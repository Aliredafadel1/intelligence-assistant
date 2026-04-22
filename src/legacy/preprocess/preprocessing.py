from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .preprocess_common import project_root
    from .preprocess_ml import run_preprocess_ml
    from .preprocess_rag import run_preprocess_rag
except ImportError:
    from preprocess_common import project_root
    from preprocess_ml import run_preprocess_ml
    from preprocess_rag import run_preprocess_rag


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run both preprocessing flows: ML cleaned table + RAG Q/A pairs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_for_prediction.csv",
        help="Input CSV path forwarded to ML and RAG preprocessors.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Output directory used for ML and RAG generated files.",
    )
    parser.add_argument(
        "--inbound-only",
        action="store_true",
        help="Forwarded to ML preprocessing only.",
    )
    parser.add_argument(
        "--drop-duplicate-clean-text",
        action="store_true",
        help="Forwarded to both preprocessors.",
    )
    parser.add_argument(
        "--min-clean-text-length",
        type=int,
        default=1,
        help="Forwarded to both preprocessors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    ml_output = out_dir / "full_tweets_for_prediction_clean.csv"
    rag_output = out_dir / "rag_qa_pairs_clean.csv"

    print("Running ML preprocessing...")
    _, cleaned, ml_output_path = run_preprocess_ml(
        input_path=args.input,
        output_path=ml_output,
        inbound_only=args.inbound_only,
        drop_duplicate_clean_text=args.drop_duplicate_clean_text,
        min_clean_text_length=args.min_clean_text_length,
    )

    print("Running RAG preprocessing...")
    _, _, rag_pairs, rag_output_path = run_preprocess_rag(
        input_path=args.input,
        output_path=rag_output,
        drop_duplicate_clean_text=args.drop_duplicate_clean_text,
        min_clean_text_length=args.min_clean_text_length,
    )
    print(f"Done. ML rows: {len(cleaned)} -> {ml_output_path}")
    print(f"Done. RAG pairs: {len(rag_pairs)} -> {rag_output_path}")


if __name__ == "__main__":
    main()
