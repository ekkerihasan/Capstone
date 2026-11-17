"""Stage 6.3 training: ROI crop -> 3D classifier, tumor present / absent.

    python -m src.train_cls --data-root <msd_or_phantom_root> --roi-size 48,48,32 \
                            --epochs 30 --out results/review2_demo

Oracle path: the ROI is cut with the *ground-truth* mask, so classifier error is
measured apart from segmentation error. When stage 6.2 is trained, `roi.py` takes
the predicted mask instead and nothing else here changes.

Every epoch writes checkpoints/stage63/last.pt and the loop resumes from it with
--resume (overnight lab-PC access is not guaranteed). best.pt tracks the highest
validation AUROC, with validation loss as a tie-break.

Negatives: real MSD Task07 has a tumor in *every* case, so a case-level label has
one class. `--sampling patch` (use it for real MSD) learns tumor vs non-tumor
*tissue* instead: per case one jittered tumor window and `--neg-per-case`
tumor-free windows from the same pancreas (roi.sample_patch_starts). Rows in
cls_predictions.csv are then patches, named <case>#pos / <case>#neg<k>; the split
stays patient-wise, so no patient's tissue sits on both sides of calibration.
`--sampling case` (default) is the original one-ROI-per-case path, kept for the
phantom, which does have tumor-free cases. Decision: docs/FINAL_2DAY_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
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
from .roi import bbox_fits, crop_at, crop_roi, sample_patch_starts
from .train_seg import get_split
from .transforms import build_transforms

STAGE = "stage63"
ROI_SIZE = (48, 48, 32)       # CPU-friendly default; the 8 GB GPU run uses --roi-size 96


def selection_score(val: dict, train_loss: float, has_val: bool) -> tuple:
    """Sort key for "is this the best epoch so far", larger is better.

    (val AUROC, -val loss). AUROC first because it is what stage 6.3 reports and
    what survives a model being badly calibrated; the negated loss breaks ties
    between the many epochs that reach the same AUROC on a small validation set.
    Undefined AUROC (a single-class group, or no validation group at all) drops
    to -inf so any epoch with a real AUROC outranks it.
    """
    if not has_val:
        return (float("-inf"), -float(train_loss))
    auroc = float(val["auroc"])
    if np.isnan(auroc):
        return (float("-inf"), -float(val["loss"]))
    return (auroc, -float(val["loss"]))


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


def _case_rng(seed: int, cid: str) -> np.random.Generator:
    """Per-case generator: the same case yields the same patches in train_cls
    and predict_cls, whatever order the cases arrive in."""
    return np.random.default_rng([seed, zlib.crc32(cid.encode())])


def load_patch_items(cases, group: str, roi_size, n_neg: int = 2, mask_fn=None,
                     seed: int = SEED) -> list:
    """Patch-level items (see module docstring). Same dict layout as
    load_roi_cases plus 'patient', so stack / predict_table / MC Dropout and the
    stage 6.4 CSV contract are unchanged.

    Tissue for negatives comes from `mask_fn`'s mask when given (the real path:
    predicted pancreas), but the tumor that negatives must avoid, and the
    positive's location, always come from the ground truth.
    """
    tf = build_transforms(train=False)
    items = []
    for case in cases:
        cid = case_id(case["image"])
        data = tf(case)
        image = np.asarray(data["image"], dtype=np.float32)[0]
        label = np.asarray(data["label"])[0]
        truth = tumor_stats(label, SPACING_MM)
        tissue = label if mask_fn is None else np.asarray(mask_fn(case, label))
        combined = (tissue >= 1).astype(np.uint8)
        combined[label == 2] = 2
        picks = sample_patch_starts(combined, roi_size, n_neg, _case_rng(seed, cid))
        if not picks:
            print(f"[6.3] skip {cid}: no pancreas tissue to sample")
            continue
        if truth["has_tumor"] and len(picks) - 1 < n_neg:
            print(f"[6.3] {cid}: only {len(picks) - 1}/{n_neg} tumor-free windows fit")
        k = 0
        for starts, y in picks:
            name = f"{cid}#pos" if y else f"{cid}#neg{k}"
            k += 0 if y else 1
            items.append({
                "case": name, "patient": cid, "group": group,
                "x": crop_at(image, starts, roi_size)[None].astype(np.float32),
                "y_true": int(y),
                "eq_diam_cm": float(truth["eq_diam_cm"]) if y else 0.0,
            })
    return items


def load_items(args_or_cfg: dict, cases, group, roi_size, mask_fn=None) -> list:
    """Dispatch on the sampling mode stored in the CLI args / checkpoint."""
    if args_or_cfg.get("sampling", "case") == "patch":
        return load_patch_items(cases, group, roi_size, args_or_cfg.get("neg_per_case", 2),
                                mask_fn=mask_fn, seed=args_or_cfg.get("seed", SEED))
    return load_roi_cases(cases, group, roi_size, args_or_cfg.get("margin_mm", 10.0), mask_fn)


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
    p.add_argument("--sampling", choices=["case", "patch"], default="case",
                   help="'patch' for real MSD (every case has a tumor); see module docstring")
    p.add_argument("--neg-per-case", type=int, default=2, help="tumor-free windows per case (patch mode)")
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
    print(f"[6.3] cutting {args.sampling} ROIs {roi_size} from the ground-truth mask ...")
    cfg = {"sampling": args.sampling, "neg_per_case": args.neg_per_case, "seed": args.seed,
           "margin_mm": args.margin_mm}
    train_items = load_items(cfg, split["train"], "train", roi_size)
    val_items = load_items(cfg, split["val"], "val", roi_size)
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
    extra = {"model": args.model, "roi_size": list(roi_size), "dropout": args.dropout, **cfg}

    for epoch in range(start, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, xtr, ytr, optimizer, device, weight, args.batch)
        val = evaluate(model, xva, yva, device, weight) if len(yva) else {"loss": float("nan"), "auroc": float("nan")}
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val["loss"], 5),
               "val_auroc": None if np.isnan(val["auroc"]) else round(val["auroc"], 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        # best.pt is selected on val AUROC, with val loss only as a tie-break.
        #
        # Selecting on val loss alone saved a chance-level epoch: on a small
        # validation set the loss explodes once the model grows confident, so
        # its minimum sits early, before the model has learned anything, while
        # AUROC keeps climbing. AUROC is also the metric this stage reports, and
        # a checkpoint should be chosen by the number it will be judged on.
        # A single-class validation group makes AUROC undefined, so that case
        # falls back to the loss.
        score = selection_score(val, train_loss, has_val=bool(len(yva)))
        improved = best is None or score > tuple(best)
        if improved:
            best = list(score)
        save_checkpoint(ckpt_dir / LAST, model, epoch, optimizer, best_metric=best,
                        history=history, extra=extra)
        if improved:
            save_checkpoint(ckpt_dir / BEST, model, epoch, best_metric=best,
                            history=history, extra=extra)
        print(f"[6.3] epoch {epoch:3d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} "
              f"| val AUROC {row['val_auroc']} | {row['seconds']}s{' *best*' if improved else ''}")

    (out / "cls_training_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    if history:
        plot_curves(history, out / "fig_stage63_training.png", args.stamp)
    print(f"[6.3] done. best checkpoint: {ckpt_dir / BEST}")


if __name__ == "__main__":
    main()
