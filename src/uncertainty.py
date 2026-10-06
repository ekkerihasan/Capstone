"""Stage 6.4, part one: Monte Carlo Dropout (Gal & Ghahramani, ICML 2016).

At inference the Dropout layers are put back into training mode and the
network is run T times. The mean of the T softmax outputs is the probability
the rest of the pipeline uses; the spread across passes is the epistemic
uncertainty that stage 6.6 turns into an abstention.

BatchNorm stays in eval mode: only Dropout is stochastic.

Caveat pinned by tests/test_train_seg.py: MONAI only passes the `dropout`
constructor argument to the convolution blocks (the Attention U-Net gates keep
p = 0.0), so `enable_dropout` sets p on the modules explicitly instead of
trusting the constructor.

Signatures frozen in docs/REVIEW2_TEMP_PLAN.md section 9.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd

MC_SAMPLES = 20     # T in CLAUDE.md


def enable_dropout(model: nn.Module, p: float | None = None) -> list:
    """Put every dropout module (Dropout, Dropout3d, AlphaDropout, ...) into
    train mode while the rest of the model stays in eval mode. With `p` given,
    every one of them is set to that probability. Returns the names of the
    modules switched on, so a caller can assert it was not a no-op."""
    model.eval()
    names = []
    for name, m in model.named_modules():
        if isinstance(m, _DropoutNd):
            if p is not None:
                m.p = float(p)
            m.train()
            names.append(name or type(m).__name__)
    return names


def disable_dropout(model: nn.Module) -> None:
    model.eval()


def predictive_entropy(probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x, T: int = MC_SAMPLES, device=None,
                       p: float | None = None) -> dict:
    """T stochastic forward passes over one batch.

    Returns numpy arrays:
        mean      [N, C]   mean softmax over the T passes — the calibrated probability
        std       [N, C]   standard deviation of the softmax across passes
        samples   [T, N, C]
        entropy   [N]      predictive entropy of the mean
    `p` overrides the dropout probability of every dropout module for this call
    (see the module docstring for why that is necessary with MONAI networks).
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    x = torch.as_tensor(x)
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    model.to(device)
    switched = enable_dropout(model, p)
    try:
        samples = torch.stack([torch.softmax(model(x), dim=1) for _ in range(T)])   # T,N,C
    finally:
        disable_dropout(model)
    mean = samples.mean(0)
    out = {
        "mean": mean.cpu().numpy(),
        "std": samples.std(0, unbiased=False).cpu().numpy(),
        "samples": samples.cpu().numpy(),
        "entropy": predictive_entropy(mean.cpu().numpy()),
        "dropout_modules": switched,
    }
    return out


@torch.no_grad()
def mc_dropout_table(model: nn.Module, items, T: int = MC_SAMPLES, device=None,
                     batch: int = 8, p: float | None = None) -> list:
    """Convenience for predict_cls / run_demo_pipeline: `items` is a list of
    dicts with at least {'case', 'x' (tensor [C,D,H,W]), 'y_true'} and
    optionally 'group', 'eq_diam_cm'. Returns one row per case in the
    cls_predictions.csv layout plus the MC columns:

        case, group, y_true, prob_tumor, eq_diam_cm, mc_std, entropy
    """
    rows = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([torch.as_tensor(it["x"]) for it in chunk])
        res = mc_dropout_predict(model, x, T=T, device=device, p=p)
        for j, it in enumerate(chunk):
            rows.append({
                "case": it["case"],
                "group": it.get("group", ""),
                "y_true": int(it["y_true"]),
                "prob_tumor": float(res["mean"][j, 1]),
                "eq_diam_cm": float(it.get("eq_diam_cm", 0.0)),
                "mc_std": float(res["std"][j, 1]),
                "entropy": float(res["entropy"][j]),
            })
    return rows
