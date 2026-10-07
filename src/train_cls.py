"""Stage 6.3 training: ROI crop -> 3D classifier, tumor present / absent.

    python -m src.train_cls --data-root <msd_or_phantom_root> --roi-size 48,48,32 \
                            --epochs 30 --out results/review2_demo

Oracle path: the ROI is cut with the *ground-truth* mask, so classifier error is
measured apart from segmentation error. When stage 6.2 is trained, `roi.py` takes
the predicted mask instead and nothing else here changes.

Every epoch writes checkpoints/stage63/last.pt and the loop resumes from it with
--resume (overnight lab-PC access is not guaranteed). best.pt tracks the lowest
validation loss.

Negatives: real MSD Task07 has a tumor in *every* case, so a classifier trained on
it alone sees one class. The phantom supplies negatives; real data needs PanTS
`tumor? = 0` cases or non-tumor crops, which is still an open decision
(docs/REVIEW2_TEMP_PLAN.md section 6). A single-class train set is warned about
rather than silently trained on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import set_determinism

from .checkpoints import BEST, LAST, resume_if_possible, save_checkpoint, stage_dir
from .classifier import build_classifier, describe, tumor_probability
from .config import CHECKPOINT_DIR, MSD_DIR, RESULTS_DIR, SEED, SPACING_MM, parse_patch
from .data import assert_disjoint, case_id
from .metrics import auroc, tumor_stats
from .roi import bbox_fits, crop_roi
from .train_seg import get_split
from .transforms import build_transforms

STAGE = "stage63"
ROI_SIZE = (48, 48, 32)       # CPU-friendly default; the 8 GB GPU run uses --roi-size 96


def load_roi_cases(cases, group: str, roi_size, margin_mm: float = 10.0, mask_fn=None) -> list:
    """Preprocess each case with the stage 6.1 transforms and cut its ROI.

    Returns [{case, group, x [1,*roi_size] float32, y_true, eq_diam_cm}, ...].
    `mask_fn(case_dict, label_array) -> mask` swaps the ground-truth mask for a
    predicted one (the real path); the default is the ground truth.
    A case whose mask has no pancreas is skipped with a warning, never cropped.
    """
    tf = build_transforms(train=False)
    items = []
    for case in cases:
        data = tf(case)
        image = np.asarray(data["image"], dtype=np.float32)[0]
        label = np.asarray(data["label"])[0]
        truth = tumor_stats(label, SPACING_MM)
        mask = label if mask_fn is None else mask_fn(case, label)
        try:
            roi = crop_roi(image, mask, out_size=roi_size, margin_mm=margin_mm)
        except ValueError as e:
            print(f"[6.3] skip {case_id(case['image'])}: {e}")
            continue
        if not bbox_fits(roi["bbox"], roi_size):
            print(f"[6.3] {case_id(case['image'])}: pancreas box exceeds roi_size {tuple(roi_size)}, truncated")
        items.append({
            "case": case_id(case["image"]),
            "group": group,
            "x": roi["image"][None].astype(np.float32),
            "y_true": int(truth["has_tumor"]),
            "eq_diam_cm": float(truth["eq_diam_cm"]),
        })
    return items


def stack(items):
    x = torch.from_numpy(np.stack([it["x"] for it in items]))
    y = torch.tensor([it["y_true"] for it in items], dtype=torch.long)
    return x, y


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random flip per spatial axis, independently per batch element."""
    out = x.clone()
    for i in range(out.shape[0]):
        for axis in (1, 2, 3):
            if torch.rand(()) < 0.5:
                out[i] = out[i].flip(axis)
    return out


def class_weights(y: torch.Tensor, device) -> torch.Tensor | None:
    counts = torch.bincount(y, minlength=2).float()
    if (counts == 0).any():
        print(f"[6.3] WARNING train set has class counts {counts.tolist()} - single class, "
              "the classifier cannot learn present/absent (see module docstring)")
        return None
    return (counts.sum() / (2.0 * counts)).to(device)


@torch.no_grad()
def evaluate(model, x, y, device, weight=None, batch=8):
    """Loss and AUROC over a set of ROIs. AUROC is NaN when only one class is present."""
    model.eval()
    probs, loss, n = [], 0.0, len(y)
    for i in range(0, n, batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        logits = model(xb)
        loss += float(F.cross_entropy(logits, yb, weight=weight, reduction="sum"))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy() if probs else np.array([])
    both = n > 0 and len(set(y.tolist())) == 2
    return {"loss": loss / max(n, 1), "auroc": auroc(y.numpy(), p) if both else float("nan")}


def train_one_epoch(model, x, y, optimizer, device, weight, batch, aug=True):
    model.train()
    order = torch.randperm(len(y))
    total, steps = 0.0, 0
    for i in range(0, len(y), batch):
        idx = order[i:i + batch]
        if len(idx) < 2:                      # BatchNorm cannot train on a single sample
            continue
        xb, yb = x[idx], y[idx]
        if aug:
            xb = augment(xb)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb, weight=weight)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def plot_curves(history, path, stamp: bool):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Stage 6.3 classifier training")
    ax.legend()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if stamp:
        from .phantom import stamp_figure
        stamp_figure(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_argparser():
    p = argparse.ArgumentParser(description="Stage 6.3 ROI classifier training")
    p.add_argument("--data-root", default=str(MSD_DIR))
    p.add_argument("--split", default="", help="split JSON to reuse (created if missing)")
    p.add_argument("--out", default=str(RESULTS_DIR / "review2"))
    p.add_argument("--ckpt-root", default=str(CHECKPOINT_DIR))
    p.add_argument("--model", default="densenet121")
    p.add_argument("--roi-size", default="48,48,32", help="'96' or '48,48,32'")
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--stamp", action="store_true", help="mark figures as phantom-derived")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_determinism(args.seed)
    device = torch.device(args.device)
    roi_size = parse_patch(args.roi_size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir(STAGE, root=args.ckpt_root)

    split = get_split(args)
    assert_disjoint(split)
    print(f"[6.3] cutting ROIs {roi_size} from the ground-truth mask ...")
    train_items = load_roi_cases(split["train"], "train", roi_size, args.margin_mm)
    val_items = load_roi_cases(split["val"], "val", roi_size, args.margin_mm)
    if not train_items:
        raise SystemExit("[6.3] no usable training ROIs")
    xtr, ytr = stack(train_items)
    xva, yva = stack(val_items) if val_items else (torch.empty(0), torch.empty(0, dtype=torch.long))
    print(f"[6.3] train={len(ytr)} (tumor={int(ytr.sum())}) val={len(yva)} (tumor={int(yva.sum())}) "
          f"device={device}")

    model = build_classifier(args.model, dropout=args.dropout).to(device)
    print("[6.3]", describe(model))
    weight = class_weights(ytr, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start, best, history = (0, None, [])
    if args.resume:
        start, best, history = resume_if_possible(STAGE, model, optimizer, root=args.ckpt_root,
                                                  map_location=device)
    extra = {"model": args.model, "roi_size": list(roi_size), "dropout": args.dropout,
             "margin_mm": args.margin_mm}

    for epoch in range(start, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, xtr, ytr, optimizer, device, weight, args.batch)
        val = evaluate(model, xva, yva, device, weight) if len(yva) else {"loss": float("nan"), "auroc": float("nan")}
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val["loss"], 5),
               "val_auroc": None if np.isnan(val["auroc"]) else round(val["auroc"], 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        score = val["loss"] if len(yva) else train_loss     # lower is better
        improved = best is None or score < best
        if improved:
            best = score
        save_checkpoint(ckpt_dir / LAST, model, epoch, optimizer, best_metric=best,
                        history=history, extra=extra)
        if improved:
            save_checkpoint(ckpt_dir / BEST, model, epoch, best_metric=best,
                            history=history, extra=extra)
        print(f"[6.3] epoch {epoch:3d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} "
              f"| val AUROC {row['val_auroc']} | {row['seconds']}s{' *' if improved else ''}")

    (out / "cls_training_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    if history:
        plot_curves(history, out / "fig_stage63_training.png", args.stamp)
    print(f"[6.3] done. best checkpoint: {ckpt_dir / BEST}")


if __name__ == "__main__":
    main()
