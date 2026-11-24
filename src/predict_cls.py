"""Stage 6.3 inference: write the one CSV that crosses into stage 6.4.

    python -m src.predict_cls --split results/review2_demo/split_6cases.json \
                              --data-root <root> --out results/review2_demo [--mc 20]

Writes <out>/cls_predictions.csv with columns

    case, group, y_true, prob_tumor, eq_diam_cm        (+ mc_std, entropy with --mc)

for every group in the split (train, val, calib). Stage 6.4 calibrates on `calib`
and tests on `val`; `train` rows are kept so over-fitting is visible, and must
never be used for calibration. With --mc T, `prob_tumor` is the MC-Dropout mean
over T passes and requires a checkpoint trained with --dropout > 0.

Also prints AUROC and sensitivity/specificity overall and per size bucket
(metrics.by_size_bucket, as CLAUDE.md requires).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism

from .checkpoints import BEST, load_checkpoint
from .classifier import build_classifier, tumor_probability
from .config import CHECKPOINT_DIR, MSD_DIR, RESULTS_DIR, SEED
from .data import case_id, load_split, rebase_split_to_msd
from .metrics import auroc, by_size_bucket, sensitivity_specificity
from .train_cls import STAGE, load_items, stack
from .uncertainty import MC_SAMPLES, mc_dropout_table

COLUMNS = ["case", "group", "y_true", "prob_tumor", "eq_diam_cm"]


def predict_table(model, items, device, mc: int = 0, batch: int = 8) -> pd.DataFrame:
    """One row per case (or per patch, for a --sampling patch checkpoint) in the
    cls_predictions.csv layout."""
    if not items:
        return pd.DataFrame(columns=COLUMNS)
    if mc:
        return pd.DataFrame(mc_dropout_table(model, items, T=mc, device=device, batch=batch))
    x, _ = stack(items)
    probs = torch.cat([tumor_probability(model, x[i:i + batch].to(device)).cpu()
                       for i in range(0, len(items), batch)]).numpy()
    return pd.DataFrame({
        "case": [it["case"] for it in items],
        "group": [it["group"] for it in items],
        "y_true": [it["y_true"] for it in items],
        "prob_tumor": probs.astype(float),
        "eq_diam_cm": [it["eq_diam_cm"] for it in items],
    })


def report(df: pd.DataFrame, group: str) -> None:
    sub = df[df["group"] == group]
    if sub.empty:
        return
    y, p = sub["y_true"].to_numpy(), sub["prob_tumor"].to_numpy()
    line = f"[6.3] {group}: n={len(sub)} tumor={int(y.sum())}"
    if len(set(y.tolist())) == 2:
        sens, spec = sensitivity_specificity(y, (p >= 0.5).astype(int))
        line += f" | AUROC {auroc(y, p):.3f} | sens {sens:.3f} | spec {spec:.3f}"
    print(line)
    sub = sub.assign(correct=((p >= 0.5).astype(int) == y).astype(float))
    print(by_size_bucket(sub, ["correct"]).to_string())


def build_argparser():
    p = argparse.ArgumentParser(description="Stage 6.3 predictions -> cls_predictions.csv")
    p.add_argument("--split", required=True, help="split JSON written by train_cls / train_seg")
    p.add_argument("--data-root", default=str(MSD_DIR),
                   help="rebase the split's paths onto this dataset root")
    p.add_argument("--out", default=str(RESULTS_DIR / "review2"))
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / STAGE / BEST))
    p.add_argument("--mc", type=int, default=0, const=MC_SAMPLES, nargs="?",
                   help="MC-Dropout passes (bare --mc = 20); needs a --dropout > 0 checkpoint")
    p.add_argument("--pred-masks", default="",
                   help="dir of <case>.npy masks from predict_seg --save-masks: the real path, "
                        "ROIs cut from the stage 6.2 prediction instead of the ground truth")
    p.add_argument("--groups", nargs="*", default=None, help="default: every group in the split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def pred_mask_fn(mask_dir):
    """mask_fn reading <case>.npy files written by predict_seg --save-masks."""
    root = Path(mask_dir)

    def mask_fn(case, label):
        pred = np.load(root / f"{case_id(case['image'])}.npy")
        if pred.shape != label.shape:
            raise SystemExit(f"[6.3] {root} mask {pred.shape} != grid {label.shape}: "
                             "masks must come from predict_seg on the same 6.1 transforms")
        return pred
    return mask_fn


def load_cls_checkpoint(path, device="cpu"):
    """Classifier + its training config, as train_cls saved them."""
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt["extra"]
    model = build_classifier(cfg["model"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_determinism(SEED)
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = ckpt["extra"]
    model = build_classifier(cfg["model"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.mc and cfg["dropout"] <= 0:
        raise SystemExit("[6.3] --mc needs a checkpoint trained with --dropout > 0; this one has none")
    roi_size = tuple(cfg["roi_size"])

    split = rebase_split_to_msd(load_split(args.split), args.data_root)

    mask_fn = pred_mask_fn(args.pred_masks) if args.pred_masks else None
    if mask_fn is not None:
        print(f"[6.3] ROIs from predicted masks in {args.pred_masks}")

    frames = []
    for group in (args.groups or list(split)):
        if mask_fn is not None:
            missing = [c for c in split[group]
                       if not (Path(args.pred_masks) / f"{case_id(c['image'])}.npy").exists()]
            if missing:
                print(f"[6.3] {group}: no predicted mask for {len(missing)} cases, group skipped")
                continue
        items = load_items(cfg, split[group], group, roi_size, mask_fn=mask_fn)
        frames.append(predict_table(model, items, device, mc=args.mc))
    df = pd.concat(frames, ignore_index=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "cls_predictions.csv"
    df.to_csv(dest, index=False)
    print(f"[6.3] wrote {len(df)} rows -> {dest}")
    for group in df["group"].unique():
        report(df, group)


if __name__ == "__main__":
    main()
