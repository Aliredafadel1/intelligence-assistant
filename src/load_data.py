from __future__ import annotations

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_raw_path() -> Path:
    return project_root() / "data" / "raw" / "twcs.csv"


def default_sample_path() -> Path:
    return project_root() / "data" / "sample" / "sample.csv"


def load_twcs(
    path: str | Path | None = None,
    *,
    use_sample: bool = False,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """
    Load the Twitter Customer Support CSV into a DataFrame.

    Parameters
    ----------
    path:
        Explicit file path. If None, uses ``data/raw/twcs.csv`` or the sample file.
    use_sample:
        If True and ``path`` is None, loads ``data/sample/sample.csv``.
    read_csv_kwargs:
        Forwarded to :func:`pandas.read_csv` (e.g. ``nrows=10_000`` for a quick slice).
    """
    if path is not None:
        p = Path(path)
    elif use_sample:
        p = default_sample_path()
    else:
        p = default_raw_path()

    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    return pd.read_csv(p, **read_csv_kwargs)