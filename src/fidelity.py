"""Stage 6.5, part two: the F-Fidelity masking test.

A saliency map is a claim: *these* voxels are why the model said what it said.
The claim is testable. Mask the most salient fraction of the volume and the
class score should fall sharply; mask the same fraction at random and it should
barely move. If it does not, the map is decoration, and the pipeline should not
show it to a clinician.

    drop_topk     score(x) - score(x with the top k% most salient voxels masked)
    drop_random   the same, averaged over `n_random` random masks of size k%
    margin        drop_topk - mean(drop_random)
    passed        mean margin over all k > max(z * mean std, MIN_MARGIN)

The verdict is the mean margin across every k rather than a vote per k. At a
small k neither mask moves a saturated softmax at all, so both drops are zero:
that k carries no evidence either way, and counting it as a failure would sink
a map that is perfectly faithful at the k where masking actually bites. The
absolute floor MIN_MARGIN stops a zero margin passing when the random baseline
happens to have zero variance.

Comparing against random masks rather than against an absolute threshold is the
point: masking *anything* perturbs the score, so the only meaningful question is
whether masking what the map pointed at perturbs it more than masking noise.

**Scope.** The published F-Fidelity protocol (Zheng et al., ICLR 2025) also
fine-tunes the model on randomly masked inputs first, so that a masked volume is
not out of distribution for it. That is out of scope for Phase-II, which defines
this stage as "top-k mask vs random mask, measure output drop". The consequence
is honest and worth stating in the report: part of any drop measured here is
distribution shift from the masking itself, which is exactly what the random
baseline controls for, and why the test is a comparison rather than a number.
"""
from __future__ import annotations

import numpy as np
import torch

K_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
N_RANDOM = 5
Z = 1.0             # how many random-baseline std devs the top-k drop must clear
MIN_MARGIN = 0.01   # ...and at least this much class probability, in absolute terms


def mask_top_k(x, saliency, k_frac: float, baseline: float = 0.0):
    """Replace the k_frac most salient voxels with `baseline`.

    The baseline is 0.0 because stage 6.1 normalises the HU window to [0, 1],
    so 0 is the bottom of the window rather than an arbitrary value.
    """
    x = torch.as_tensor(x).clone()
    sal = np.asarray(saliency).ravel()
    n = max(1, int(round(k_frac * sal.size)))
    idx = np.argpartition(-sal, n - 1)[:n]
    flat = x.view(x.shape[0], x.shape[1], -1)
    flat[..., torch.as_tensor(idx, dtype=torch.long)] = baseline
    return flat.view_as(x)


def mask_random(x, k_frac: float, baseline: float = 0.0, rng=None):
    """Replace a random k_frac of voxels with `baseline` - the control."""
    rng = np.random.default_rng() if rng is None else rng
    x = torch.as_tensor(x).clone()
    size = int(np.prod(x.shape[2:]))
    n = max(1, int(round(k_frac * size)))
    idx = rng.choice(size, size=n, replace=False)
    flat = x.view(x.shape[0], x.shape[1], -1)
    flat[..., torch.as_tensor(idx, dtype=torch.long)] = baseline
    return flat.view_as(x)


@torch.no_grad()
def _score(model, x, class_idx: int) -> float:
    return float(torch.softmax(model(torch.as_tensor(x)), dim=1)[0, class_idx])


def fidelity_test(model, x, saliency, class_idx: int | None = None,
                  k_fractions=K_FRACTIONS, n_random: int = N_RANDOM,
                  baseline: float = 0.0, z: float = Z, seed: int = 42,
                  min_margin: float = MIN_MARGIN) -> dict:
    """Run the masking comparison across several k and return a verdict.

    The verdict is a single pass/fail because stage 6.6 needs one: a map that
    fails is a reason to escalate the case rather than show the explanation.
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        logits = model(torch.as_tensor(x))
        if class_idx is None:
            class_idx = int(logits[0].argmax())
        base = float(torch.softmax(logits, dim=1)[0, class_idx])

    k_fractions = tuple(k_fractions)
    sal = np.asarray(saliency, dtype=float)
    expected_size = int(np.prod(torch.as_tensor(x).shape[2:]))
    if not sal.size:
        reason = "saliency map is empty"
    elif not np.isfinite(sal).all():
        reason = "saliency map contains non-finite values"
    elif sal.size != expected_size:
        reason = "saliency map shape does not match the input"
    elif float(np.ptp(sal)) <= 1e-12:
        reason = "saliency map is constant"
    else:
        reason = None
    if reason is not None:
        return {
            "class_idx": int(class_idx),
            "base_score": base,
            "curve": [],
            "mean_margin": 0.0,
            "threshold": float(min_margin),
            "passed": False,
            "n_k": len(k_fractions),
            "n_k_passed": 0,
            "z": float(z),
            "min_margin": float(min_margin),
            "valid_saliency": False,
            "failure_reason": reason,
        }

    k_fractions = tuple(k_fractions)
    rows = []
    for k in k_fractions:
        top = base - _score(model, mask_top_k(x, saliency, k, baseline), class_idx)
        rand = [base - _score(model, mask_random(x, k, baseline, rng), class_idx)
                for _ in range(n_random)]
        rand_mean, rand_std = float(np.mean(rand)), float(np.std(rand))
        rows.append({
            "k_frac": float(k),
            "drop_topk": float(top),
            "drop_random_mean": rand_mean,
            "drop_random_std": rand_std,
            "margin": float(top - rand_mean),
            "passed": bool(top - rand_mean > max(z * rand_std, min_margin)),
        })

    mean_margin = float(np.mean([r["margin"] for r in rows]))
    pooled_std = float(np.mean([r["drop_random_std"] for r in rows]))
    threshold = max(z * pooled_std, min_margin)
    return {
        "class_idx": int(class_idx),
        "base_score": base,
        "curve": rows,
        "mean_margin": mean_margin,
        "threshold": threshold,
        "passed": bool(mean_margin > threshold),
        "n_k": len(rows),
        "n_k_passed": int(sum(r["passed"] for r in rows)),
        "z": float(z),
        "min_margin": float(min_margin),
        "valid_saliency": True,
        "failure_reason": None,
    }


def plot_fidelity(result: dict, path=None, title: str = "", stamp: bool = True):
    """Deletion curve: the top-k drop against the random control, per k."""
    import matplotlib.pyplot as plt

    from .phantom import stamp_figure
    from .viz import save_fig

    curve = result["curve"]
    ks = [r["k_frac"] * 100 for r in curve]
    top = [r["drop_topk"] for r in curve]
    rnd = [r["drop_random_mean"] for r in curve]
    err = [r["drop_random_std"] for r in curve]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, top, "o-", color="#d1495b", label="mask most salient")
    ax.errorbar(ks, rnd, yerr=err, fmt="s--", color="#5b4fbf", capsize=3,
                label=f"mask at random (n={len(curve) and N_RANDOM})")
    ax.set_xlabel("% of voxels masked")
    ax.set_ylabel("drop in class probability")
    verdict = "PASS" if result["passed"] else "FAIL"
    ax.set_title(f"{title} F-Fidelity: {verdict} "
                 f"(mean margin {result['mean_margin']:+.3f})", fontsize=11)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    if stamp:
        stamp_figure(fig)
    if path:
        save_fig(fig, path)
        plt.close(fig)
    return fig
