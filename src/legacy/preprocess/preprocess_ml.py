from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .preprocess_common import clean_full_dataset, project_root
except ImportError:
    from preprocess_common import clean_full_dataset, project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Prepare ML-ready cleaned dataset from tweet data."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (overrides output filename location).",
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
        default=root / "data" / "processed" / "full_tweets_for_prediction_clean.csv",
        help="Output path for cleaned ML dataset.",
    )
    parser.add_argument(
        "--inbound-only",
        action="store_true",
        help="Keep only inbound rows.",
    )
    parser.add_argument(
        "--drop-duplicate-clean-text",
        action="store_true",
        help="Drop rows with duplicate clean_text after standard deduplication.",
    )
    parser.add_argument(
        "--min-clean-text-length",
        type=int,
        default=1,
        help="Minimum clean_text length to keep a row.",
    )
    return parser.parse_args()


def run_preprocess_ml(
    *,
    input_path: Path,
    output_path: Path,
    inbound_only: bool,
    drop_duplicate_clean_text: bool,
    min_clean_text_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    cleaned = clean_full_dataset(
        raw,
        inbound_only=inbound_only,
        drop_duplicate_clean_text=drop_duplicate_clean_text,
        min_clean_text_length=min_clean_text_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return raw, cleaned, output_path


def main() -> None:
    args = parse_args()
    output_path = (
        (args.out_dir / args.output.name) if args.out_dir is not None else args.output
    )
    raw, cleaned, resolved_output_path = run_preprocess_ml(
        input_path=args.input,
        output_path=output_path,
        inbound_only=args.inbound_only,
        drop_duplicate_clean_text=args.drop_duplicate_clean_text,
        min_clean_text_length=args.min_clean_text_length,
    )

    print(f"Input rows                : {len(raw)}")
    print(f"Clean rows                : {len(cleaned)}")
    print(f"Saved ML dataset          : {resolved_output_path}")


if __name__ == "__main__":
    main()
