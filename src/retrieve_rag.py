from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .index_rag import default_index_dir
except ImportError:
    from index_rag import default_index_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def read_env_file_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    prefix = f"{key}="
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        return value
    return None


def is_placeholder_key(value: str) -> bool:
    return "YOUR_NEW_OPENAI_API_KEY" in value


def resolve_openai_api_key() -> str:
    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    file_key = (read_env_file_value(ENV_PATH, "OPENAI_API_KEY") or "").strip()
    if file_key and not is_placeholder_key(file_key):
        key = file_key
    else:
        key = env_key

    if not key:
        raise EnvironmentError("OPENAI_API_KEY is missing. Set it in .env or environment.")
    if is_placeholder_key(key):
        raise EnvironmentError("OPENAI_API_KEY is placeholder text. Replace it with a real key.")
    if not key.startswith("sk-") or len(key) < 40:
        raise EnvironmentError(
            "OPENAI_API_KEY format looks invalid. It should start with 'sk-' and be full length."
        )
    return key


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

    if backend == "openai":
        embeddings_path = root / "openai_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError("OpenAI embeddings artifact missing for openai backend.")
        payload = {
            "embeddings": np.load(embeddings_path),
            "meta": meta,
        }
        return backend, manifest, payload

    raise ValueError(f"Unsupported backend in index metadata: {backend}")


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
    elif backend == "openai":
        api_key = resolve_openai_api_key()
        embeddings = payload["embeddings"]
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings matrix is empty.")
        model = payload["meta"].get("embedding_model", "text-embedding-3-small")
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model, input=[q])
        q_vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        sims = _cosine_dense(q_vec, embeddings)
    else:
        raise ValueError("backend must be either 'tfidf' or 'openai'")

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
    parser = argparse.ArgumentParser(
        description="Query RAG index (tfidf/openai backend) and return top-k similar entries."
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
        "--index-dir",
        type=Path,
        default=None,
        help="Index directory (default: data/processed/rag_index).",
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
