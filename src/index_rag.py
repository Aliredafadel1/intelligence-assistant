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

try:
    from .prepare_datasets import default_processed_dir
except ImportError:
    from prepare_datasets import default_processed_dir

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

    # Prefer .env if it has a non-placeholder key; otherwise fallback to process env.
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


def default_index_dir() -> Path:
    return default_processed_dir() / "rag_index"


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


def fit_tfidf_index(texts: list[str]) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def embed_texts_openai(
    texts: list[str],
    *,
    model: str,
    batch_size: int = 128,
) -> np.ndarray:
    api_key = resolve_openai_api_key()

    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("OpenAI embeddings returned unexpected shape.")
    return arr


def save_rag_index_tfidf(
    manifest: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: sparse.csr_matrix,
    out_dir: Path,
) -> dict[str, Any]:
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


def save_rag_index_openai(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: Path,
    *,
    model: str,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    embeddings_path = out_dir / "openai_embeddings.npy"
    meta_path = out_dir / "index_meta.json"

    manifest.to_csv(manifest_path, index=False)
    np.save(embeddings_path, embeddings)

    meta = {
        "backend": "openai",
        "embedding_model": model,
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


def build_rag_index_from_corpus(
    corpus: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    backend: str = "tfidf",
    openai_model: str = "text-embedding-3-small",
    max_chunk_chars: int = 0,
    chunk_overlap: int = 50,
    concat_query: bool = False,
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
    if backend == "openai":
        embeddings = embed_texts_openai(texts, model=openai_model)
        return save_rag_index_openai(manifest, embeddings, target_dir, model=openai_model)
    raise ValueError("backend must be either 'tfidf' or 'openai'")


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    default_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    parser = argparse.ArgumentParser(
        description="Build RAG index from retrieval_corpus.csv (tfidf or openai embeddings)."
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
        choices=("tfidf", "openai"),
        default="tfidf",
        help="Index backend to build.",
    )
    parser.add_argument(
        "--openai-model",
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
        openai_model=args.openai_model,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        concat_query=args.concat_query,
    )
    print(f"Backend         : {args.backend}")
    print(f"Indexed rows    : {paths['n_rows']:,}")
    print(f"Feature columns : {paths['n_features']:,}")
    if args.backend == "tfidf":
        for k in ("manifest", "vectorizer", "matrix", "meta"):
            print(f"{k:14s} : {paths[k]}")
    else:
        for k in ("manifest", "embeddings", "meta"):
            print(f"{k:14s} : {paths[k]}")
if __name__ == "__main__":
    main()
