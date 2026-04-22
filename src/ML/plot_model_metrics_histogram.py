from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ..preprocess_common import project_root
except ImportError:
    from preprocess_common import project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    artifacts_dir = root / "data" / "artifacts"
    parser = argparse.ArgumentParser(
        description="Plot histogram-style bar charts for MAE, RMSE, and R2 across models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=artifacts_dir / "priority_model_regression_metrics.json",
        help="Path to the model regression metrics JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifacts_dir / "priority_model_metrics_histogram.png",
        help="Path to save the generated histogram image.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("leaderboard_by_rmse", [])
    if not rows:
        raise ValueError("No rows found in leaderboard_by_rmse.")

    model_names = [str(row.get("model", "unknown")) for row in rows]
    mae_vals = [float(row.get("mae", 0.0)) for row in rows]
    rmse_vals = [float(row.get("rmse", 0.0)) for row in rows]
    r2_vals = [float(row.get("r2", 0.0)) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Model Comparison: MAE vs RMSE vs R2", fontsize=13)

    axes[0].bar(model_names, mae_vals, color="#4c78a8")
    axes[0].set_title("MAE")
    axes[0].set_ylabel("Error (lower is better)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(model_names, rmse_vals, color="#f58518")
    axes[1].set_title("RMSE")
    axes[1].set_ylabel("Error (lower is better)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(model_names, r2_vals, color="#54a24b")
    axes[2].set_title("R2")
    axes[2].set_ylabel("Score (higher is better)")
    axes[2].tick_params(axis="x", rotation=20)

    for ax, vals in zip(axes, [mae_vals, rmse_vals, r2_vals]):
        for idx, value in enumerate(vals):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram image: {output_path}")


if __name__ == "__main__":
    main()
