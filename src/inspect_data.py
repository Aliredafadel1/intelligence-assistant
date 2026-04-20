from pathlib import Path
import pandas as pd


def load_sample(path: str | Path, nrows: int = 100_000) -> pd.DataFrame:
    """
    Load a sample of the dataset for quick inspection.
    """
    return pd.read_csv(path, nrows=nrows)


def inspect_shape(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)


def inspect_columns(df: pd.DataFrame) -> None:
    print("\nColumns:")
    print(df.columns.tolist())


def inspect_dtypes(df: pd.DataFrame) -> None:
    print("\nData types:")
    print(df.dtypes)


def inspect_missing(df: pd.DataFrame) -> None:
    print("\nMissing values:")
    print(df.isnull().sum())


def inspect_duplicates(df: pd.DataFrame) -> None:
    print("\nDuplicate rows:", df.duplicated().sum())


def inspect_memory(df: pd.DataFrame) -> None:
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nEstimated memory usage: {memory_mb:.2f} MB")


def inspect_inbound(df: pd.DataFrame) -> None:
    if "inbound" in df.columns:
        print("\nInbound distribution:")
        print(df["inbound"].value_counts(dropna=False))
    else:
        print("\nColumn 'inbound' not found.")


def inspect_text_sample(df: pd.DataFrame, sample_size: int = 5) -> None:
    if "text" not in df.columns:
        print("\nColumn 'text' not found.")
        return

    text_series = df["text"].dropna()

    if text_series.empty:
        print("\nNo non-null text values found.")
        return

    sample_size = min(sample_size, len(text_series))
    print("\nSample text:")
    for i, text in enumerate(text_series.sample(sample_size, random_state=42), start=1):
        print(f"{i}. {text}")


def inspect_text_length(df: pd.DataFrame) -> None:
    if "text" not in df.columns:
        print("\nColumn 'text' not found.")
        return

    text_lengths = df["text"].fillna("").str.len()

    print("\nText length summary:")
    print(text_lengths.describe())


def run_full_inspection(path: str | Path, nrows: int = 100_000) -> pd.DataFrame:
    """
    Run all inspection steps on a sample of the dataset.
    """
    df = load_sample(path, nrows=nrows)

    print("=" * 60)
    print("DATA INSPECTION REPORT")
    print("=" * 60)

    inspect_shape(df)
    inspect_columns(df)
    inspect_dtypes(df)
    inspect_missing(df)
    inspect_duplicates(df)
    inspect_memory(df)
    inspect_inbound(df)
    inspect_text_sample(df)
    inspect_text_length(df)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    return df


if __name__ == "__main__":
    dataset_path = Path("data/sample/sample.csv")
    run_full_inspection(dataset_path)