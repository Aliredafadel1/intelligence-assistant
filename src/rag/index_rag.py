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
import numpy as np
import pandas as pd
from chromadb.config import Settings
from openai import OpenAI

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

try:
    from ..prepare_datasets import default_processed_dir
except ImportError:
    from prepare_datasets import default_processed_dir

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


def default_index_dir() -> Path:
    return default_processed_dir() / "rag_index"


def default_chroma_dir() -> Path:
    return default_processed_dir() / "chroma_db"


def _build_chroma_client(chroma_path: Path) -> chromadb.ClientAPI:
    host = (os.environ.get("CHROMA_HOST") or "").strip()
    port_raw = (os.environ.get("CHROMA_PORT") or "").strip()
    if host:
        try:
            port = int(port_raw) if port_raw else 8000
        except ValueError:
            port = 8000
        return chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
    return chromadb.PersistentClient(
        path=str(chroma_path.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Character-based chunking with overlap. If ``max_chars`` <= 0, returns ``[text]``.
    """
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def build_index_manifest(
    corpus: pd.DataFrame,
    *,
    max_chunk_chars: int = 0,
    chunk_overlap: int = 50,
    concat_query: bool = False,
) -> pd.DataFrame:
    """
    One row per indexed vector row. ``index_text`` is what gets embedded / vectorized.
    """
    required = {"query_text", "document_text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, r in corpus.iterrows():
        doc = r["document_text"]
        if doc is None or (isinstance(doc, float) and pd.isna(doc)):
            continue
        doc_s = str(doc).strip()
        if not doc_s:
            continue

        q = r["query_text"]
        q_s = "" if q is None or (isinstance(q, float) and pd.isna(q)) else str(q).strip()

        base = {k: r[k] for k in r.index}
        parts = chunk_text(doc_s, max_chunk_chars, chunk_overlap)
        for i, part in enumerate(parts):
            row = dict(base)
            row["chunk_index"] = i
            if concat_query and q_s:
                row["index_text"] = f"{q_s}\n\n{part}"
            else:
                row["index_text"] = part
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def fit_tfidf_index(texts: list[str]) -> tuple[object, object]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "TF-IDF dependencies are missing. Install training dependencies for tfidf backend."
        ) from exc
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def _get_openai_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
    return OpenAI(api_key=api_key)


def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return arr / norms


def embed_texts_openai(
    texts: list[str],
    *,
    model: str,
    batch_size: int = 128,
) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for embedding.")
    client = _get_openai_client()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=chunk)
        all_vectors.extend(item.embedding for item in response.data)
    arr = _normalize_embeddings(np.array(all_vectors, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError("OpenAI embeddings returned unexpected shape.")
    return arr


def save_rag_index_tfidf(
    manifest: pd.DataFrame,
    vectorizer: object,
    matrix: object,
    out_dir: Path,
) -> dict[str, Any]:
    try:
        import joblib
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "TF-IDF dependencies are missing. Install training dependencies for tfidf backend."
        ) from exc
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    vectorizer_path = out_dir / "tfidf_vectorizer.joblib"
    matrix_path = out_dir / "tfidf_matrix.npz"
    meta_path = out_dir / "index_meta.json"

    manifest.to_csv(manifest_path, index=False)
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, matrix)

    meta = {
        "backend": "tfidf",
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "index_text_column": "index_text",
        "manifest_csv": str(manifest_path.as_posix()),
        "vectorizer_path": str(vectorizer_path.as_posix()),
        "matrix_path": str(matrix_path.as_posix()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "vectorizer": str(vectorizer_path),
        "matrix": str(matrix_path),
        "meta": str(meta_path),
        "n_rows": meta["n_rows"],
        "n_features": meta["n_features"],
    }


def save_rag_index_embeddings(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    *,
    model: str,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    embeddings_path = out_dir / "local_embeddings.npy"
    meta_path = out_dir / "index_meta.json"

    manifest.to_csv(manifest_path, index=False)
    np.save(embeddings_path, embeddings)

    meta = {
        "backend": "openai",
        "embedding_model": model,
        "embedding_provider": "openai",
        "n_rows": int(embeddings.shape[0]),
        "n_features": int(embeddings.shape[1]),
        "index_text_column": "index_text",
        "manifest_csv": str(manifest_path.as_posix()),
        "embeddings_path": str(embeddings_path.as_posix()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "embeddings": str(embeddings_path),
        "meta": str(meta_path),
        "n_rows": meta["n_rows"],
        "n_features": meta["n_features"],
    }


def _to_chroma_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _build_chroma_records(manifest: pd.DataFrame, embeddings: np.ndarray) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for idx, row in manifest.iterrows():
        qid = _to_chroma_scalar(row.get("question_tweet_id"))
        aid = _to_chroma_scalar(row.get("answer_tweet_id"))
        chunk_idx = _to_chroma_scalar(row.get("chunk_index"))
        record_id = f"{qid or 'q'}:{aid or 'a'}:{chunk_idx or 0}:{idx}"
        ids.append(record_id)
        documents.append(str(row.get("index_text", "")))

        md: dict[str, str | int | float | bool] = {}
        for col, value in row.items():
            if col == "index_text":
                continue
            clean = _to_chroma_scalar(value)
            if clean is not None:
                md[col] = clean
        metadatas.append(md)

    if len(ids) != embeddings.shape[0]:
        raise ValueError("Manifest row count does not match embedding count for Chroma upsert.")
    return ids, documents, metadatas


def upsert_chroma_index(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    chroma_path: Path,
    collection_name: str,
    embedding_model: str,
    embedding_provider: str = "openai",
    batch_size: int = 200,
) -> dict[str, Any]:
    chroma_path = chroma_path.resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = _build_chroma_client(chroma_path)
    # Keep collection metadata minimal for compatibility with older Chroma server builds.
    collection = client.get_or_create_collection(name=collection_name)

    ids, documents, metadatas = _build_chroma_records(manifest, embeddings)
    if not ids:
        raise ValueError("No IDs prepared for Chroma upsert.")
    empty_docs = sum(1 for d in documents if not str(d).strip())
    if empty_docs > 0:
        raise ValueError(f"Found {empty_docs} empty documents; aborting Chroma upsert.")
    vectors = embeddings.tolist()

    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=vectors[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    # Debug checks: verify persistence and basic retrieval immediately after ingestion.
    count = collection.count()
    print(f"Chroma count    : {count}")
    print(f"DB path exists  : {chroma_path.exists()}")
    sample_results = collection.query(
        query_embeddings=[vectors[0]],
        n_results=min(3, max(1, count)),
        include=["documents", "distances"],
    )
    print(f"Retrieved docs  : {sample_results.get('documents')}")

    return {
        "chroma_path": str(chroma_path),
        "chroma_collection": collection_name,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "n_rows": len(ids),
        "n_features": int(embeddings.shape[1]),
    }


def build_rag_index_from_corpus(
    corpus: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    backend: str = "openai",
    storage: str = "local",
    embed_model: str = "text-embedding-3-small",
    max_chunk_chars: int = 0,
    chunk_overlap: int = 50,
    concat_query: bool = False,
    chroma_path: Path | None = None,
    chroma_collection: str = "rag_tickets",
    chroma_batch_size: int = 200,
) -> dict[str, Any]:
    manifest = build_index_manifest(
        corpus,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        concat_query=concat_query,
    )
    if manifest.empty:
        raise ValueError("No rows to index: corpus is empty after filtering.")

    texts = manifest["index_text"].astype("string").fillna("").tolist()
    target_dir = out_dir if out_dir is not None else default_index_dir()
    if backend == "tfidf":
        vectorizer, matrix = fit_tfidf_index(texts)
        return save_rag_index_tfidf(manifest, vectorizer, matrix, target_dir)
    if backend in {"sbert", "openai"}:
        embeddings = embed_texts_openai(texts, model=embed_model)
        if storage == "local":
            return save_rag_index_embeddings(manifest, embeddings, target_dir, model=embed_model)
        if storage == "chroma":
            cpath = chroma_path if chroma_path is not None else default_chroma_dir()
            result = upsert_chroma_index(
                manifest,
                embeddings,
                chroma_path=cpath,
                collection_name=chroma_collection,
                embedding_model=embed_model,
                embedding_provider="openai",
                batch_size=chroma_batch_size,
            )
            result["backend"] = "openai-chroma"
            result["embedding_provider"] = "openai"
            result["embedding_model"] = embed_model
            return result
        raise ValueError("storage must be either 'local' or 'chroma'")
    raise ValueError("backend must be either 'tfidf' or 'openai'")


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    default_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    default_collection = os.environ.get("CHROMA_COLLECTION", "rag_tickets")
    parser = argparse.ArgumentParser(
        description="Build RAG index from retrieval_corpus.csv (tfidf or OpenAI embeddings)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "retrieval_corpus.csv",
        help="Path to retrieval_corpus.csv from prepare_datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Index output directory (default: data/processed/rag_index).",
    )
    parser.add_argument(
        "--backend",
        choices=("tfidf", "openai", "sbert"),
        default="openai",
        help="Index backend to build.",
    )
    parser.add_argument(
        "--storage",
        choices=("local", "chroma"),
        default="local",
        help="Storage target for dense embedding backend (local files or Chroma DB).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=default_model,
        help="OpenAI embedding model when backend=openai.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=0,
        help="Max characters per chunk of document_text; 0 = one chunk per row.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap when chunking (only if max-chunk-chars > 0).",
    )
    parser.add_argument(
        "--concat-query",
        action="store_true",
        help="Prepend query_text to each chunk in index_text (query-aware indexing).",
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
        "--chroma-batch-size",
        type=int,
        default=200,
        help="Batch size for Chroma upserts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    corpus = pd.read_csv(input_path)
    paths = build_rag_index_from_corpus(
        corpus,
        args.out_dir.resolve() if args.out_dir is not None else None,
        backend=args.backend,
        storage=args.storage,
        embed_model=args.embed_model,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        concat_query=args.concat_query,
        chroma_path=args.chroma_path.resolve() if args.chroma_path is not None else None,
        chroma_collection=args.chroma_collection,
        chroma_batch_size=args.chroma_batch_size,
    )
    print(f"Backend         : {args.backend}")
    print(f"Storage         : {args.storage}")
    if args.backend in {"sbert", "openai"}:
        print("Embed provider  : openai")
        print(f"Embed model     : {args.embed_model}")
    print(f"Indexed rows    : {paths['n_rows']:,}")
    print(f"Feature columns : {paths['n_features']:,}")
    if args.backend == "tfidf":
        for k in ("manifest", "vectorizer", "matrix", "meta"):
            print(f"{k:14s} : {paths[k]}")
    else:
        if args.storage == "local":
            for k in ("manifest", "embeddings", "meta"):
                print(f"{k:14s} : {paths[k]}")
        else:
            for k in ("chroma_path", "chroma_collection"):
                print(f"{k:14s} : {paths[k]}")
if __name__ == "__main__":
    main()
