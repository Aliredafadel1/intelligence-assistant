from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .preprocess_common import build_rag_qa_pairs, clean_full_dataset, project_root
except ImportError:
    from preprocess_common import build_rag_qa_pairs, clean_full_dataset, project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Prepare clean RAG Q/A pairs from tweet data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_for_prediction.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "rag_qa_pairs_clean.csv",
        help="Output path for clean RAG pairs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (overrides output filename location).",
    )
    parser.add_argument(
        "--drop-duplicate-clean-text",
        action="store_true",
        help="Drop duplicate clean_text rows before pairing.",
    )
    parser.add_argument(
        "--min-clean-text-length",
        type=int,
        default=1,
        help="Minimum clean_text length to keep a row.",
    )
    return parser.parse_args()


def run_preprocess_rag(
    *,
    input_path: Path,
    output_path: Path,
    drop_duplicate_clean_text: bool,
    min_clean_text_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    cleaned = clean_full_dataset(
        raw,
        inbound_only=False,
        drop_duplicate_clean_text=drop_duplicate_clean_text,
        min_clean_text_length=min_clean_text_length,
    )
    rag_pairs = build_rag_qa_pairs(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rag_pairs.to_csv(output_path, index=False)
    return raw, cleaned, rag_pairs, output_path


def main() -> None:
    args = parse_args()
    output_path = (
        (args.out_dir / args.output.name) if args.out_dir is not None else args.output
    )
    raw, cleaned, rag_pairs, resolved_output_path = run_preprocess_rag(
        input_path=args.input,
        output_path=output_path,
        drop_duplicate_clean_text=args.drop_duplicate_clean_text,
        min_clean_text_length=args.min_clean_text_length,
    )

    print(f"Input rows                : {len(raw)}")
    print(f"Clean rows (pairing base) : {len(cleaned)}")
    print(f"RAG Q&A pairs             : {len(rag_pairs)}")
    print(f"Saved RAG dataset         : {resolved_output_path}")


if __name__ == "__main__":
    main()
