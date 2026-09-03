"""Metrics shared by every stage.

CLAUDE.md rule: results are always reported overall AND for the sub-2 cm bucket,
where tumor size is the equivalent spherical diameter derived from mask volume.
"""
from __future__ import annotations

import numpy as np

from .config import LABEL_PANCREAS, LABEL_TUMOR, SMALL_TUMOR_CM


def dice(pred, gt) -> float:
    """Dice for two boolean masks. Two empty masks agree perfectly -> 1.0."""
    a = np.asarray(pred).astype(bool)
    b = np.asarray(gt).astype(bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def voxel_volume_ml(spacing_mm) -> float:
    """Volume of one voxel in millilitres (1 mL = 1 cm^3 = 1000 mm^3)."""
    return float(np.prod(spacing_mm) / 1000.0)


def equivalent_spherical_diameter_cm(volume_ml: float) -> float:
    """Diameter of the sphere with the same volume: d = (6V/pi)^(1/3), V in cm^3."""
    if volume_ml <= 0:
        return 0.0
    return float((6.0 * volume_ml / np.pi) ** (1.0 / 3.0))


def tumor_stats(label, spacing_mm, tumor_label=LABEL_TUMOR, pancreas_label=LABEL_PANCREAS):
    """Per-case size description used for the sub-2 cm stratum."""
    mask = np.asarray(label).squeeze()
    vox_ml = voxel_volume_ml(spacing_mm)
    tumor_vox = int((mask == tumor_label).sum())
    pancreas_vox = int((mask >= pancreas_label).sum())
    tumor_ml = tumor_vox * vox_ml
    diam = equivalent_spherical_diameter_cm(tumor_ml)
    return {
        "pancreas_vox": pancreas_vox,
        "tumor_vox": tumor_vox,
        "pancreas_ml": round(pancreas_vox * vox_ml, 3),
        "tumor_ml": round(tumor_ml, 3),
        "eq_diam_cm": round(diam, 3),
        "has_tumor": tumor_vox > 0,
        "sub2cm": bool(0 < diam < SMALL_TUMOR_CM),
    }


def size_bucket(eq_diam_cm: float, threshold: float = SMALL_TUMOR_CM) -> str:
    if eq_diam_cm <= 0:
        return "no_tumor"
    return "sub2cm" if eq_diam_cm < threshold else "ge2cm"


def by_size_bucket(df, value_cols, diam_col: str = "eq_diam_cm"):
    """Mean of `value_cols` overall and per size bucket — the table shape every
    review deck asks for. `df` is a pandas DataFrame with one row per case."""
    import pandas as pd

    buckets = df[diam_col].map(size_bucket)
    rows = {"overall": df[value_cols].mean()}
    for name, group in df.groupby(buckets):
        rows[name] = group[value_cols].mean()
    out = pd.DataFrame(rows).T
    out["n"] = [len(df)] + [int((buckets == k).sum()) for k in out.index[1:]]
    return out.round(3)
