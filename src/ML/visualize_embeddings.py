from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize RAG embeddings with PCA or t-SNE (2D scatter)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/rag_index/manifest.csv"),
        help="Path to manifest CSV produced during local embedding index build.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/rag_index/local_embeddings.npy"),
        help="Path to local embeddings .npy file.",
    )
    parser.add_argument(
        "--method",
        choices=("pca", "tsne"),
        default="pca",
        help="Projection method.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Optional max rows used for plotting (for t-SNE speed).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="brand_author_id",
        help="Column name in manifest used for point coloring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/embedding_projection.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def _ensure_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib and scikit-learn are required. Install via requirements/train.txt."
        ) from exc


def _project(embeddings: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA  # pyright: ignore[reportMissingImports]

        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    from sklearn.manifold import TSNE  # pyright: ignore[reportMissingImports]

    return TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        max_iter=1000,
    ).fit_transform(embeddings)


def main() -> None:
    _ensure_dependencies()
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    args = parse_args()
    manifest_path = args.manifest.resolve()
    emb_path = args.embeddings.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(emb_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape={embeddings.shape}")
    if len(manifest) != embeddings.shape[0]:
        raise ValueError("manifest row count must match embeddings row count")

    n = min(int(args.sample_size), embeddings.shape[0]) if args.sample_size > 0 else embeddings.shape[0]
    picked_idx = manifest.sample(n=n, random_state=42).index.to_numpy()
    sampled = manifest.loc[picked_idx].copy().reset_index(drop=True)
    sampled_emb = embeddings[picked_idx]
    coords = _project(sampled_emb, args.method)

    label_col = args.label_col if args.label_col in sampled.columns else None
    labels = sampled[label_col].astype("string").fillna("unknown") if label_col else pd.Series(["all"] * len(sampled))
    top_labels = labels.value_counts().head(8).index.tolist()
    plot_labels = labels.where(labels.isin(top_labels), "other")

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(plot_labels.astype("string").unique().tolist())
    cmap = plt.get_cmap("tab10")
    for idx, value in enumerate(unique_labels):
        mask = plot_labels == value
        ax.scatter(
            coords[mask.to_numpy(), 0],
            coords[mask.to_numpy(), 1],
            s=10,
            alpha=0.65,
            label=value,
            color=cmap(idx % 10),
        )

    ax.set_title(f"Embedding Projection ({args.method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding projection: {output_path}")


if __name__ == "__main__":
    main()
