from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.null.NullTelemetry")
os.environ.setdefault("POSTHOG_DISABLED", "1")

import chromadb
import joblib
import numpy as np
import pandas as pd
from chromadb.config import Settings
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

try:
    from .index_rag import default_index_dir
except ImportError:
    from index_rag import default_index_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ENV_PATH)


def embed_query(model: str, query: str) -> np.ndarray:
    q = query.strip()
    if not q:
        raise ValueError("Query is empty.")
    encoder = SentenceTransformer(model)
    vec = encoder.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return vec[0]


def _cosine_dense(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec.astype(np.float32)
    m = matrix.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros((m.shape[0],), dtype=np.float32)
    m_norm = np.linalg.norm(m, axis=1)
    denom = q_norm * m_norm
    denom = np.where(denom == 0, 1e-12, denom)
    return (m @ q) / denom


def default_chroma_dir() -> Path:
    return default_index_dir().parent / "chroma_db"


def load_rag_index(index_dir: Path) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    root = index_dir.resolve()
    meta_path = root / "index_meta.json"
    manifest_path = root / "manifest.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Required index artifact not found: {meta_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Required index artifact not found: {manifest_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    backend = meta.get("backend", "tfidf")
    if backend == "tfidf":
        vectorizer_path = root / "tfidf_vectorizer.joblib"
        matrix_path = root / "tfidf_matrix.npz"
        if not vectorizer_path.exists() or not matrix_path.exists():
            raise FileNotFoundError("TF-IDF artifacts missing for tfidf backend.")
        payload: dict[str, Any] = {
            "vectorizer": joblib.load(vectorizer_path),
            "matrix": sparse.load_npz(matrix_path),
            "meta": meta,
        }
        return backend, manifest, payload

    if backend in {"openai", "sbert"}:
        embeddings_path = root / "local_embeddings.npy"
        if backend == "openai" and not embeddings_path.exists():
            embeddings_path = root / "openai_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError("Dense embeddings artifact missing for sbert/openai backend.")
        payload = {
            "embeddings": np.load(embeddings_path),
            "meta": meta,
        }
        normalized_backend = "sbert" if backend == "openai" else backend
        return normalized_backend, manifest, payload

    raise ValueError(f"Unsupported backend in index metadata: {backend}")


def retrieve_top_k_chroma(
    query: str,
    *,
    k: int,
    collection_name: str,
    chroma_path: Path,
    model: str | None,
) -> pd.DataFrame:
    q = query.strip()
    if not q:
        raise ValueError("Query is empty.")
    db = chromadb.PersistentClient(
        path=str(chroma_path.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = db.get_collection(name=collection_name)
    metadata = collection.metadata or {}
    effective_model = model or metadata.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    if model is not None and metadata.get("embedding_model") and model != metadata.get("embedding_model"):
        print(
            f"Warning: requested embed model '{model}' differs from collection model "
            f"'{metadata.get('embedding_model')}'. Using collection model."
        )
        effective_model = metadata.get("embedding_model")
    q_vec = embed_query(effective_model, q).tolist()
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    rows: list[dict[str, Any]] = []
    for i, row_id in enumerate(ids):
        md = metas[i] if i < len(metas) and metas[i] is not None else {}
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
        score = 1.0 - dist
        row = {
            "rank": i + 1,
            "similarity_score": score,
            "id": row_id,
            "index_text": docs[i] if i < len(docs) else "",
        }
        if isinstance(md, dict):
            row.update(md)
        rows.append(row)
    return pd.DataFrame(rows)


def retrieve_top_k(
    query: str,
    backend: str,
    manifest: pd.DataFrame,
    payload: dict[str, Any],
    *,
    k: int = 5,
) -> pd.DataFrame:
    q = query.strip()
    if not q:
        raise ValueError("Query is empty.")
    if backend == "tfidf":
        vectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
        if matrix.shape[0] == 0:
            raise ValueError("Index matrix is empty.")
        query_vec = vectorizer.transform([q])
        sims = cosine_similarity(query_vec, matrix).ravel()
    elif backend == "sbert":
        embeddings = payload["embeddings"]
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings matrix is empty.")
        model = payload["meta"].get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        q_vec = embed_query(model, q)
        sims = _cosine_dense(q_vec, embeddings)
    else:
        raise ValueError("backend must be either 'tfidf' or 'sbert'")

    if sims.size == 0:
        return manifest.head(0).copy()

    take = max(1, min(k, sims.size))
    top_idx = sims.argsort()[::-1][:take]
    out = manifest.iloc[top_idx].copy().reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(1, "similarity_score", sims[top_idx])
    return out


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "rank",
        "similarity_score",
        "question_tweet_id",
        "answer_tweet_id",
        "query_text",
        "document_text",
        "index_text",
        "customer_author_id",
        "brand_author_id",
        "question_created_at",
        "response_created_at",
        "chunk_index",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df.loc[:, cols]


def parse_args() -> argparse.Namespace:
    default_collection = os.environ.get("CHROMA_COLLECTION", "rag_tickets")
    default_embed_model = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    parser = argparse.ArgumentParser(
        description="Query RAG index (tfidf/sbert backend) and return top-k similar entries."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Incoming ticket text (or question) to retrieve against.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results to return.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "local", "chroma"),
        default="auto",
        help="Retrieval source: local index artifacts or Chroma DB.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Index directory (default: data/processed/rag_index).",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Persistent Chroma DB directory (default: data/processed/chroma_db).",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default=default_collection,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="SentenceTransformer embedding model used for query embedding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save top-k rows as CSV.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of a plain table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "chroma":
        chroma_path = args.chroma_path.resolve() if args.chroma_path is not None else default_chroma_dir()
        topk = retrieve_top_k_chroma(
            args.query,
            k=args.k,
            collection_name=args.chroma_collection,
            chroma_path=chroma_path,
            model=args.embed_model,
        )
        backend = "sbert-chroma"
        topk = select_output_columns(topk)
    else:
        index_dir = args.index_dir.resolve() if args.index_dir is not None else default_index_dir()
        backend, manifest, payload = load_rag_index(index_dir)
        topk = retrieve_top_k(args.query, backend, manifest, payload, k=args.k)
        topk = select_output_columns(topk)

    if args.output is not None:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        topk.to_csv(out_path, index=False)
        print(f"Saved retrieval results: {out_path}")

    if args.json:
        print(json.dumps(topk.to_dict(orient="records"), indent=2, default=str))
    else:
        print(f"Backend: {backend}")
        if topk.empty:
            print("No retrieval results.")
        else:
            table = topk.to_string(index=False)
            print(table.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
