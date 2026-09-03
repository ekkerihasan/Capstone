"""Stage 6.2 — score a trained checkpoint and draw the prediction overlays.

    python -m src.predict_seg --checkpoint checkpoints/stage62/best.pt \
                              --split results/review1/split_281cases.json \
                              --group val --out results/review2

Writes to --out:
    seg_per_case.csv         per-case Dice for pancreas and tumor, plus tumor size
    seg_by_size_bucket.csv   the same metrics overall AND for the sub-2 cm stratum
    seg_summary.json         headline numbers
    pred_<case>.png          ground truth vs prediction, side by side

The size-bucket table is the one CLAUDE.md requires in every review: a mean Dice
over all cases hides the fact that the small tumors are the ones that matter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from .checkpoints import load_checkpoint
from .config import (CHECKPOINT_DIR, LABEL_PANCREAS, LABEL_TUMOR, PATCH_SIZE,
                     RESULTS_DIR, SEED, parse_patch)
from .data import case_id, load_split
from .metrics import by_size_bucket, dice, size_bucket, tumor_stats
from .models import build_attention_unet
from .transforms import build_transforms
from .viz import best_axial_slice, save_fig


def plot_prediction(image, truth, prediction, title, path, dice_pancreas, dice_tumor):
    """Ground truth beside prediction on the same slice, contoured."""
    import matplotlib.pyplot as plt

    img = np.asarray(image).squeeze()
    gt = np.asarray(truth).squeeze()
    pr = np.asarray(prediction).squeeze()
    z = best_axial_slice(gt)
    sl = np.rot90(img[:, :, z])

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    for a, mask, label in zip(ax, (gt, pr), ("ground truth", "Attention U-Net")):
        m = np.rot90(mask[:, :, z])
        a.imshow(sl, cmap="gray")
        if (m >= LABEL_PANCREAS).any():
            a.contour(m >= LABEL_PANCREAS, colors="lime", linewidths=0.9)
        if (m == LABEL_TUMOR).any():
            a.contour(m == LABEL_TUMOR, colors="red", linewidths=0.9)
        a.set_title(label)
        a.axis("off")
    fig.suptitle(f"{title} | z={z} | Dice pancreas={dice_pancreas:.3f} tumor={dice_tumor:.3f}",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


@torch.no_grad()
def predict_case(model, case, device, patch, transform, sw_batch=2, overlap=0.25):
    """Preprocess one case, run sliding-window inference, return (image, gt, pred)
    as numpy arrays on the preprocessed grid."""
    data = transform(dict(case))
    image = data["image"].unsqueeze(0).to(device)
    truth = np.asarray(data["label"]).squeeze()
    logits = sliding_window_inference(image, patch, sw_batch, model, overlap=overlap)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy()
    return np.asarray(data["image"]).squeeze(), truth, pred, data


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 6.2 evaluation and overlays")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "stage62" / "best.pt"))
    p.add_argument("--split", required=True)
    p.add_argument("--group", default="val", help="which split group to score")
    p.add_argument("--out", default=str(RESULTS_DIR / "review2"))
    p.add_argument("--patch", default="", help="defaults to the training patch size")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-figures", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    set_determinism(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_attention_unet().to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    # The patch size is recorded in the checkpoint so evaluation matches training.
    patch = (parse_patch(args.patch) if args.patch
             else tuple(ckpt.get("extra", {}).get("patch") or PATCH_SIZE))
    print(f"[6.2-eval] {args.checkpoint} | epoch {ckpt['epoch']} | "
          f"best={ckpt.get('best_metric')} | patch={patch} | device={device}")

    cases = load_split(args.split)[args.group]
    if args.limit:
        cases = cases[: args.limit]
    transform = build_transforms(train=False)

    rows = []
    for i, case in enumerate(cases):
        name = case_id(case["image"])
        image, truth, pred, data = predict_case(model, case, device, patch, transform)

        spacing = tuple(float(x) for x in data["image"].pixdim)
        stats = tumor_stats(truth, spacing)
        row = {
            "case": name,
            "dice_pancreas": round(dice(pred >= LABEL_PANCREAS, truth >= LABEL_PANCREAS), 4),
            "dice_tumor": round(dice(pred == LABEL_TUMOR, truth == LABEL_TUMOR), 4),
            "eq_diam_cm": stats["eq_diam_cm"],
            "has_tumor": stats["has_tumor"],
            "size_bucket": size_bucket(stats["eq_diam_cm"]),
            "gt_tumor_vox": stats["tumor_vox"],
            "pred_tumor_vox": int((pred == LABEL_TUMOR).sum()),
        }
        rows.append(row)
        print(f"[6.2-eval] {name}: pancreas={row['dice_pancreas']} tumor={row['dice_tumor']} "
              f"({row['size_bucket']})")

        if i < args.max_figures:
            plot_prediction(image, truth, pred, name, out / f"pred_{name}.png",
                            row["dice_pancreas"], row["dice_tumor"])

    per_case = pd.DataFrame(rows)
    per_case.to_csv(out / "seg_per_case.csv", index=False)

    buckets = by_size_bucket(per_case, ["dice_pancreas", "dice_tumor"])
    buckets.to_csv(out / "seg_by_size_bucket.csv")
    print("\n[6.2-eval] Dice overall and by tumor size")
    print(buckets.to_string())

    with_tumor = per_case[per_case["has_tumor"]]
    summary = {
        "stage": "6.2",
        "checkpoint": str(args.checkpoint),
        "trained_epochs": ckpt["epoch"] + 1,
        "group": args.group,
        "n_cases": len(per_case),
        "mean_dice_pancreas": round(float(per_case["dice_pancreas"].mean()), 4),
        "mean_dice_tumor_over_tumor_cases": round(float(with_tumor["dice_tumor"].mean()), 4)
        if len(with_tumor) else None,
        "n_cases_with_tumor": int(len(with_tumor)),
        "n_cases_sub2cm": int((per_case["size_bucket"] == "sub2cm").sum()),
        "by_size_bucket": json.loads(buckets.to_json(orient="index")),
    }
    (out / "seg_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n[6.2-eval] mean pancreas Dice = {summary['mean_dice_pancreas']} -> {out}")
    return summary


if __name__ == "__main__":
    main()
