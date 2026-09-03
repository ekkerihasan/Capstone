"""Stage 6.2 — train the Attention U-Net for pancreas + tumor segmentation.

    python -m src.train_seg --data-root data/Task07_Pancreas \
                            --split results/review1/split_281cases.json \
                            --epochs 200 --out results/review2

Every epoch writes `checkpoints/stage62/last.pt`, so an interrupted run resumes
with `--resume` and loses at most one epoch. Metrics are appended to
`metrics_stage62.csv` after each epoch for the same reason.

Sized for the lab RTX 4060 (8 GB): 96^3 patches, batch 2, AMP on. If it OOMs,
drop `--batch 1` first, then `--patch 64` — do not shrink the network.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

from .checkpoints import BEST, LAST, resume_if_possible, save_checkpoint, stage_dir
from .config import (BATCH_SIZE, CACHE_RATE, CHECKPOINT_DIR, MSD_DIR, NUM_WORKERS,
                     RESULTS_DIR, SEED, parse_patch)
from .data import (assert_disjoint, load_msd_datalist, load_split, patient_wise_split,
                   save_split)
from .models import N_CLASSES, build_attention_unet, describe
from .transforms import build_transforms

STAGE = "stage62"
CLASS_NAMES = ("pancreas", "tumor")  # background is excluded from the metric


def get_split(args):
    """Reuse the split JSON when one exists — every stage must score the same
    held-out patients — otherwise make one and save it."""
    if args.split and Path(args.split).exists():
        split = load_split(args.split)
        print(f"[6.2] split loaded from {args.split}")
    else:
        cases = load_msd_datalist(args.data_root)
        split = patient_wise_split(cases, seed=args.seed)
        dest = Path(args.split) if args.split else Path(args.out) / f"split_{len(cases)}cases.json"
        save_split(split, dest)
        print(f"[6.2] split created -> {dest}")
    assert_disjoint(split)
    if args.limit_train:
        split["train"] = split["train"][: args.limit_train]
    if args.limit_val:
        split["val"] = split["val"][: args.limit_val]
    return split


def build_loaders(split, args, patch):
    train_tf = build_transforms(train=True, patches=True, num_samples=args.num_samples,
                                patch_size=patch)
    val_tf = build_transforms(train=False)

    train_ds = CacheDataset(split["train"], train_tf, cache_rate=args.cache_rate,
                            num_workers=args.workers, progress=False)
    val_ds = CacheDataset(split["val"], val_tf, cache_rate=args.cache_rate,
                          num_workers=args.workers, progress=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=list_data_collate,
                          pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    return train_dl, val_dl


@torch.no_grad()
def validate(model, loader, device, patch, metric, post_pred, post_label, sw_batch=2):
    """Whole-volume Dice via sliding-window inference at the training patch size."""
    model.eval()
    metric.reset()
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = sliding_window_inference(image, patch, sw_batch, model, overlap=0.25)
        preds = [post_pred(p) for p in decollate_batch(logits)]
        targets = [post_label(t) for t in decollate_batch(label)]
        metric(y_pred=preds, y=targets)
    per_class = metric.aggregate()
    metric.reset()
    return {name: float(per_class[i]) for i, name in enumerate(CLASS_NAMES)}


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp, max_steps=0):
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss = loss_fn(model(image), label)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1), steps


def plot_curves(history, path):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(df["epoch"], df["train_loss"], color="#5b4fbf")
    ax[0].set_title("Training loss (Dice + CE)")
    ax[0].set_xlabel("epoch")
    val = df.dropna(subset=["dice_pancreas"])
    if len(val):
        ax[1].plot(val["epoch"], val["dice_pancreas"], label="pancreas", color="#5b4fbf")
        ax[1].plot(val["epoch"], val["dice_tumor"], label="tumor", color="#d1495b")
        ax[1].axhline(0.75, color="grey", ls="--", lw=0.8)  # Week 1-2 pancreas target
        ax[1].set_ylim(0, 1)
        ax[1].legend()
    ax[1].set_title("Validation Dice")
    ax[1].set_xlabel("epoch")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_argparser():
    p = argparse.ArgumentParser(description="Stage 6.2 Attention U-Net training")
    p.add_argument("--data-root", default=str(MSD_DIR))
    p.add_argument("--split", default="", help="split JSON to reuse (created if missing)")
    p.add_argument("--out", default=str(RESULTS_DIR / "review2"))
    p.add_argument("--ckpt-root", default=str(CHECKPOINT_DIR))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--patch", default="96", help="'96' or '64,64,64'")
    p.add_argument("--num-samples", type=int, default=1, help="patches drawn per volume")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cache-rate", type=float, default=CACHE_RATE)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (smoke runs)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    set_determinism(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    patch = parse_patch(args.patch)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir(STAGE, root=args.ckpt_root)

    split = get_split(args)
    print(f"[6.2] train={len(split['train'])} val={len(split['val'])} | patch={patch} "
          f"batch={args.batch}x{args.num_samples} amp={use_amp} device={device}")
    train_dl, val_dl = build_loaders(split, args, patch)

    model = build_attention_unet(dropout=args.dropout).to(device)
    print("[6.2]", describe(model))
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=N_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=N_CLASSES)])

    start_epoch, best, history = 0, None, []
    if args.resume:
        start_epoch, best, history = resume_if_possible(
            STAGE, model, optimizer, scaler, scheduler, root=args.ckpt_root,
            map_location=device)
        # A scheduler state_dict carries T_max with it, so resuming a 3-epoch run
        # with --epochs 200 would keep annealing over 3 and cycle the LR back up.
        # The horizon is a property of this run, not of the checkpoint.
        if scheduler.T_max != args.epochs:
            print(f"[6.2] retargeting cosine schedule: T_max {scheduler.T_max} -> {args.epochs}")
            scheduler.T_max = args.epochs

    metrics_csv = out / "metrics_stage62.csv"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]        # the rate this epoch actually used
        train_loss, steps = train_one_epoch(model, train_dl, optimizer, loss_fn, device,
                                            scaler, use_amp, args.max_steps)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "steps": steps,
               "lr": lr, "seconds": round(time.time() - t0, 1),
               "dice_pancreas": np.nan, "dice_tumor": np.nan, "dice_mean": np.nan,
               "best": False}

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            dice = validate(model, val_dl, device, patch, metric, post_pred, post_label)
            row.update({f"dice_{k}": round(v, 4) for k, v in dice.items()})
            row["dice_mean"] = round(float(np.mean(list(dice.values()))), 4)
            if best is None or row["dice_mean"] > best:
                best = row["dice_mean"]
                row["best"] = True
                save_checkpoint(ckpt_dir / BEST, model, epoch, optimizer, scaler, scheduler,
                                best_metric=best, history=history + [row],
                                extra={"patch": list(patch), "classes": list(CLASS_NAMES)})

        history.append(row)
        # written every epoch so an interrupted run still has its metrics
        pd.DataFrame(history).to_csv(metrics_csv, index=False)
        save_checkpoint(ckpt_dir / LAST, model, epoch, optimizer, scaler, scheduler,
                        best_metric=best, history=history,
                        extra={"patch": list(patch), "classes": list(CLASS_NAMES)})

        print(f"[6.2] epoch {epoch:>3} loss={row['train_loss']:.4f} "
              f"dice_pancreas={row['dice_pancreas']} dice_tumor={row['dice_tumor']} "
              f"({row['seconds']}s){' *best*' if row['best'] else ''}")

    if history:
        plot_curves(history, out / "fig_stage62_training.png")
    summary = {
        "stage": "6.2",
        "model": "AttentionUnet",
        "epochs_completed": history[-1]["epoch"] + 1 if history else 0,
        "best_dice_mean": best,
        "patch": list(patch),
        "batch": args.batch,
        "num_samples": args.num_samples,
        "amp": use_amp,
        "device": str(device),
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "checkpoints": str(ckpt_dir),
    }
    (out / "stage62_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[6.2] done | best mean Dice = {best} -> {out}")
    return summary


if __name__ == "__main__":
    main()
