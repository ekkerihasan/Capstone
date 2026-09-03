"""Figures for review decks. Every figure is saved to results/<review>/ at
150 dpi — CLAUDE.md forbids figures that only exist inline in a notebook."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import LABEL_PANCREAS, LABEL_TUMOR

PANCREAS_COLOR = "lime"
TUMOR_COLOR = "red"
DPI = 150


def best_axial_slice(label) -> int:
    """Index of the axial slice with the most tumor, falling back to the most
    pancreas when the case has no tumor."""
    mask = np.asarray(label).squeeze()
    area = (mask == LABEL_TUMOR).sum(axis=(0, 1))
    if area.max() == 0:
        area = (mask >= LABEL_PANCREAS).sum(axis=(0, 1))
    return int(area.argmax())


def save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def plot_case(image, label, title="", path=None, show=False):
    """Three panels: preprocessed CT, colour overlay, contours."""
    import matplotlib.pyplot as plt

    img = np.asarray(image).squeeze()
    lab = np.asarray(label).squeeze()
    z = best_axial_slice(lab)
    im, lb = np.rot90(img[:, :, z]), np.rot90(lab[:, :, z])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(im, cmap="gray")
    ax[0].set_title("preprocessed CT")
    ax[1].imshow(im, cmap="gray")
    ax[1].imshow(np.ma.masked_where(lb == 0, lb), cmap="spring", alpha=0.5)
    ax[1].set_title("pancreas (1) + tumor (2)")
    ax[2].imshow(im, cmap="gray")
    ax[2].contour(lb >= LABEL_PANCREAS, colors=PANCREAS_COLOR, linewidths=0.8)
    ax[2].contour(lb == LABEL_TUMOR, colors=TUMOR_COLOR, linewidths=0.8)
    ax[2].set_title("contours: green=pancreas, red=tumor")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{title} | axial slice z={z} | shape {img.shape}", fontsize=10)
    fig.tight_layout()

    if path:
        save_fig(fig, path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_size_distribution(stats, path=None, show=False, threshold=2.0):
    """Histogram of equivalent spherical diameter with the sub-2 cm line."""
    import matplotlib.pyplot as plt

    diam = stats["eq_diam_cm"]
    with_tumor = diam[diam > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    ax[0].hist(with_tumor, bins=12, color="#5b4fbf")
    ax[0].axvline(threshold, color="r", ls="--")
    n_small = int((with_tumor < threshold).sum())
    ax[0].set_title(f"Tumor equivalent diameter — sub-{threshold:g} cm: "
                    f"{n_small}/{len(with_tumor)}")
    ax[0].set_xlabel("cm")
    ax[0].set_ylabel("cases")

    ax[1].bar(range(len(stats)), stats["pancreas_ml"], label="pancreas", color="#5b4fbf")
    ax[1].bar(range(len(stats)), stats["tumor_ml"], label="tumor", color="#d1495b")
    ax[1].set_yscale("log")
    ax[1].set_title("Annotated volume per case (mL, log)")
    ax[1].set_xlabel("case")
    ax[1].legend()

    fig.tight_layout()
    if path:
        save_fig(fig, path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
