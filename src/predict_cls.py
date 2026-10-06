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

import pandas as pd
import torch

from .checkpoints import BEST, load_checkpoint
from .classifier import build_classifier, tumor_probability
from .config import CHECKPOINT_DIR, MSD_DIR, RESULTS_DIR
from .data import load_split
from .metrics import auroc, by_size_bucket, sensitivity_specificity
from .train_cls import STAGE, load_roi_cases, stack
from .uncertainty import MC_SAMPLES, mc_dropout_table

COLUMNS = ["case", "group", "y_true", "prob_tumor", "eq_diam_cm"]


def predict_table(model, items, device, mc: int = 0, batch: int = 8) -> pd.DataFrame:
    """One row per case in the cls_predictions.csv layout."""
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
    p.add_argument("--groups", nargs="*", default=None, help="default: every group in the split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = ckpt["extra"]
    model = build_classifier(cfg["model"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.mc and cfg["dropout"] <= 0:
        raise SystemExit("[6.3] --mc needs a checkpoint trained with --dropout > 0; this one has none")
    roi_size = tuple(cfg["roi_size"])

    split = load_split(args.split, root=Path(args.data_root) / "imagesTr")
    # load_split rebases every path onto one folder; labels live in labelsTr
    for cases in split.values():
        for c in cases:
            c["label"] = str(Path(args.data_root) / "labelsTr" / Path(c["label"]).name)

    frames = []
    for group in (args.groups or list(split)):
        items = load_roi_cases(split[group], group, roi_size, cfg.get("margin_mm", 10.0))
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
