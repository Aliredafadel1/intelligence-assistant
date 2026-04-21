from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .prepare_datasets import default_processed_dir
except ImportError:
    from prepare_datasets import default_processed_dir


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


def save_rag_index(
    manifest: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: sparse.csr_matrix,
    out_dir: Path,
) -> dict[str, str]:
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


def build_rag_index_from_corpus(
    corpus: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    max_chunk_chars: int = 0,
    chunk_overlap: int = 50,
    concat_query: bool = False,
) -> dict[str, str]:
    manifest = build_index_manifest(
        corpus,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        concat_query=concat_query,
    )
    if manifest.empty:
        raise ValueError("No rows to index: corpus is empty after filtering.")

    texts = manifest["index_text"].astype("string").fillna("").tolist()
    vectorizer, matrix = fit_tfidf_index(texts)
    target_dir = out_dir if out_dir is not None else default_index_dir()
    return save_rag_index(manifest, vectorizer, matrix, target_dir)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build TF-IDF vector index from retrieval_corpus.csv (RAG indexing)."
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
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap=args.chunk_overlap,
        concat_query=args.concat_query,
    )
    print(f"Indexed rows    : {paths['n_rows']:,}")
    print(f"Feature columns : {paths['n_features']:,}")
    for k in ("manifest", "vectorizer", "matrix", "meta"):
        print(f"{k:14s} : {paths[k]}")


if __name__ == "__main__":
    main()
