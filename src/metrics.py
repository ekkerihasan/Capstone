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


# --------------------------------------------------------------- stage 6.3 / 6.4
# Classification metrics on the per-case probabilities that stage 6.3 writes to
# cls_predictions.csv (columns: case, group, y_true, prob_tumor, eq_diam_cm).
# Signatures are frozen in docs/REVIEW2_TEMP_PLAN.md section 9.


def _binary(y_true, y_pred=None):
    y = np.asarray(y_true).astype(int).ravel()
    if y_pred is None:
        return y
    return y, np.asarray(y_pred).astype(int).ravel()


def auroc(y_true, y_score) -> float:
    """Area under the ROC curve. NaN when only one class is present, because
    the ranking is then undefined — a tiny validation group can hit this."""
    y = _binary(y_true)
    s = np.asarray(y_score, dtype=float).ravel()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def confusion(y_true, y_pred):
    """2x2 array in the sklearn layout: [[TN, FP], [FN, TP]]."""
    y, p = _binary(y_true, y_pred)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def sensitivity_specificity(y_true, y_pred) -> tuple:
    """(sensitivity, specificity). Sensitivity is the tumor recall — the number
    that matters clinically, since a missed sub-2 cm tumor is the costly error.
    NaN when the denominator is empty."""
    (tn, fp), (fn, tp) = confusion(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def reliability_curve(y_true, y_prob, n_bins: int = 15) -> tuple:
    """Binned calibration of the tumor probability.

    Cases are bucketed by `y_prob` (= p(tumor)) into `n_bins` equal-width bins
    on [0, 1]. For each bin:
        bin_centre   midpoint of the bin
        accuracy     observed fraction of tumor cases in the bin
        confidence   mean predicted p(tumor) in the bin
    A calibrated model has accuracy == confidence in every bin. Empty bins are
    NaN so a plot shows the gap instead of a fake zero.

    This is the binary-classification form of the reliability diagram (the
    positive-class probability is binned, as in sklearn's calibration_curve),
    which uses the whole [0, 1] axis; the max-probability form would leave the
    lower half of the plot empty for a two-class problem.
    """
    y = _binary(y_true)
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    centre = (edges[:-1] + edges[1:]) / 2.0
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            acc[b] = y[m].mean()
            conf[b] = p[m].mean()
    return centre, acc, conf


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b| over the bins of
    `reliability_curve`. 0 is perfectly calibrated; the report asks for it next
    to AUROC because a model can rank well and still be wildly over-confident."""
    y = _binary(y_true)
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def auroc_by_size_bucket(df, y_col: str = "y_true", score_col: str = "prob_tumor",
                         diam_col: str = "eq_diam_cm"):
    """AUROC overall and per tumor-size bucket.

    `by_size_bucket` averages per-case values and is the right tool for Dice,
    ECE contributions or set sizes. AUROC is not a per-case quantity: the
    sub-2 cm AUROC is the ranking of the *sub-2 cm positives against all
    negatives*, so it needs the negatives in every bucket. Hence this helper.
    Returns a DataFrame indexed overall / sub2cm / ge2cm with columns auroc,
    n_pos, n_neg.
    """
    import pandas as pd

    y = df[y_col].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    bucket = df[diam_col].map(size_bucket).to_numpy()
    neg = y == 0
    rows = {"overall": {"auroc": auroc(y, s), "n_pos": int((y == 1).sum()), "n_neg": int(neg.sum())}}
    for name in ("sub2cm", "ge2cm"):
        keep = neg | ((y == 1) & (bucket == name))
        n_pos = int(((y == 1) & (bucket == name)).sum())
        rows[name] = {"auroc": auroc(y[keep], s[keep]) if n_pos else float("nan"),
                      "n_pos": n_pos, "n_neg": int(neg.sum())}
    return pd.DataFrame(rows).T.astype({"n_pos": int, "n_neg": int}).round(3)
