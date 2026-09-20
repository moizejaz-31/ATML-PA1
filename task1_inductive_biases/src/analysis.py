"""
Analysis utilities for Task 1.

- Cosine stability I_T
- t-SNE / UMAP visualization
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None

SEED = 6304


def cosine_stability(clean_feats, trans_feats):
    """Mean cosine sim between paired clean/transformed features."""
    cn = clean_feats / (clean_feats.norm(dim=1, keepdim=True) + 1e-8)
    tn = trans_feats / (trans_feats.norm(dim=1, keepdim=True) + 1e-8)
    return float((cn * tn).sum(dim=1).mean())


def visualize_representations(clean_f, trans_f, clean_y, trans_y,
                              class_names, model_name, transform_name,
                              method="tsne", save_path=None, seed=SEED):
    """t-SNE/UMAP of clean vs transformed features."""
    combined = np.concatenate([clean_f, trans_f])
    N = len(clean_f)
    if method == "tsne":
        proj = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(combined)
    elif method == "umap":
        assert umap, "pip install umap-learn"
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(combined)
    else:
        raise ValueError(method)

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in np.unique(np.concatenate([clean_y, trans_y])):
        name = class_names[c] if c < len(class_names) else str(c)
        mc, mt = clean_y == c, trans_y == c
        ax.scatter(proj[:N][mc, 0], proj[:N][mc, 1], marker="o", alpha=0.5, s=15, label=f"{name} (clean)")
        ax.scatter(proj[N:][mt, 0], proj[N:][mt, 1], marker="x", alpha=0.5, s=15, label=f"{name} ({transform_name})")
    ax.set_title(f"{model_name}: {method.upper()}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
