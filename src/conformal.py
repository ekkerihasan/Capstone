"""Stage 6.4, part two: split conformal prediction for the classifier.

Nonconformity of a labelled case:  s = 1 - p_hat(true class).
On the *calibration* group (the 'calib' split made in stage 6.1, 42 cases on
MSD) q_hat is the ceil((n+1)(1-alpha))/n empirical quantile of the scores. For
a new case the prediction set is every class c with 1 - p_hat(c) <= q_hat.

Guarantee (exchangeability only, nothing assumed about the model): the true
label is in the set with probability >= 1 - alpha. The set size is the honest
answer to "how sure is the model" — a set with both labels means it does not
know, and stage 6.6 abstains on exactly those cases.

Signatures frozen in docs/REVIEW2_TEMP_PLAN.md section 9.
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.1     # 90 % target coverage


def _as_probs(probs) -> np.ndarray:
    """Accept either an [N, C] probability matrix or a 1-D vector of p(tumor)
    (the `prob_tumor` column of cls_predictions.csv) and return [N, C]."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    if p.ndim != 2:
        raise ValueError(f"probs must be [N, C] or [N], got shape {p.shape}")
    return p


def nonconformity(probs, y_true) -> np.ndarray:
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=int).ravel()
    return 1.0 - p[np.arange(len(y)), y]


def calibrate(probs_calib, y_calib, alpha: float = ALPHA) -> float:
    """q_hat from the calibration group.

    q_hat is the k-th smallest calibration score with k = ceil((n+1)(1-alpha)).
    When k > n (too few calibration cases for the requested alpha) q_hat is 1.0
    and every set contains all classes — the method refuses to promise what the
    data cannot support, which is the right failure mode for a medical tool.
    With 42 calibration cases, alpha = 0.1 gives k = 39, so it is supported.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(nonconformity(probs_calib, y_calib))
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))     # the k-th smallest score
    if k > n:
        return 1.0
    return float(scores[k - 1])


def predict_set(probs, qhat: float) -> list:
    """One sorted list of class labels per case: c is in the set when
    1 - p(c) <= q_hat."""
    p = _as_probs(probs)
    keep = (1.0 - p) <= qhat
    return [[int(c) for c in np.flatnonzero(row)] for row in keep]


def coverage(sets, y_true) -> float:
    """Fraction of cases whose true label is inside the set."""
    y = np.asarray(y_true, dtype=int).ravel()
    if len(y) == 0:
        return float("nan")
    hits = [int(t) in s for s, t in zip(sets, y)]
    return float(np.mean(hits))


def set_sizes(sets) -> np.ndarray:
    return np.asarray([len(s) for s in sets], dtype=int)


def summarize_sets(sets, y_true) -> dict:
    """Coverage plus the abstention statistics stage 6.6 will act on."""
    size = set_sizes(sets)
    n = len(size)
    return {
        "n": int(n),
        "coverage": coverage(sets, y_true),
        "mean_set_size": float(size.mean()) if n else float("nan"),
        "frac_singleton": float((size == 1).mean()) if n else float("nan"),
        "frac_abstain": float((size != 1).mean()) if n else float("nan"),   # empty or full
        "frac_empty": float((size == 0).mean()) if n else float("nan"),
    }


def coverage_curve(probs_calib, y_calib, probs_test, y_test, alphas=None) -> list:
    """Empirical coverage and mean set size on the test group over a sweep of
    alpha, each calibrated on the calibration group. Feeds the stage 6.4
    coverage figure: a method that works sits on or above the diagonal."""
    alphas = np.linspace(0.02, 0.5, 25) if alphas is None else np.asarray(alphas, dtype=float)
    rows = []
    for a in alphas:
        q = calibrate(probs_calib, y_calib, alpha=float(a))
        sets = predict_set(probs_test, q)
        rows.append({"alpha": float(a), "target_coverage": float(1.0 - a), "q_hat": q,
                     "coverage": coverage(sets, y_test),
                     "mean_set_size": float(set_sizes(sets).mean()) if len(sets) else float("nan")})
    return rows
