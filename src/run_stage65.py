"""Stage 6.5 in batch: Grad-CAM++ and the F-Fidelity masking test over every
patch of one split group, on the trained stage 6.3 classifier.

    python -m src.run_stage65 --split results/review1_local/full281/split_281cases.json \
        --data-root data/Task07_Pancreas --ckpt checkpoints/stage63/best.pt \
        --pred-masks results/final/masks --group val --out results/final/stage65

The app shows one saliency map per click; this produces the evidence for the
report - how often a map is faithful, split by true class and by tumor size:

    stage65_fidelity.csv     one row per patch: prob, predicted class, fidelity verdict
    stage65_summary.json     pass rates overall / tumor patches / tissue patches / sub-2 cm
    fig_saliency_<patch>.png Grad-CAM++ on the most confident tumor patches

Patches come from the same sampler as stage 6.3 (`--sampling patch` checkpoints)
so the maps explain exactly the decisions the classifier was evaluated on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MSD_DIR, RESULTS_DIR
from .data import load_split, rebase_split_to_msd
from .explain import gradcam_plusplus, overlay_saliency
from .fidelity import fidelity_test
from .metrics import size_bucket
from .predict_cls import load_cls_checkpoint, pred_mask_fn
from .train_cls import load_items


def explain_items(model, items, device="cpu", k_fractions=None, n_random=None) -> pd.DataFrame:
    """Grad-CAM++ and F-Fidelity on each item; the predicted class is explained."""
    kw = {}
    if k_fractions is not None:
        kw["k_fractions"] = k_fractions
    if n_random is not None:
        kw["n_random"] = n_random
    rows, cams = [], {}
    for it in items:
        x = torch.as_tensor(it["x"])[None].to(device)
        cam, cls, _ = gradcam_plusplus(model, x)
        fid = fidelity_test(model, x, cam, class_idx=cls, **kw)
        with torch.no_grad():
            prob = float(torch.softmax(model(x), dim=1)[0, 1])
        rows.append({
            "case": it["case"], "group": it.get("group", ""), "y_true": int(it["y_true"]),
            "eq_diam_cm": float(it.get("eq_diam_cm", 0.0)), "prob_tumor": prob,
            "pred_class": cls, "base_score": fid["base_score"], "mean_margin": fid["mean_margin"],
            "threshold": fid["threshold"], "fidelity_pass": bool(fid["passed"]),
        })
        cams[it["case"]] = cam
    return pd.DataFrame(rows), cams


def summarize(df: pd.DataFrame) -> dict:
    def rate(sub):
        return {"n": int(len(sub)),
                "pass_rate": round(float(sub["fidelity_pass"].mean()), 4) if len(sub) else None}

    buckets = df["eq_diam_cm"].map(size_bucket)
    return {
        "overall": rate(df),
        "tumor_patches": rate(df[df["y_true"] == 1]),
        "tissue_patches": rate(df[df["y_true"] == 0]),
        "sub2cm_tumor_patches": rate(df[buckets == "sub2cm"]),
        "ge2cm_tumor_patches": rate(df[buckets == "ge2cm"]),
        "correct_predictions": rate(df[df["pred_class"] == df["y_true"]]),
        "wrong_predictions": rate(df[df["pred_class"] != df["y_true"]]),
    }


def build_argparser():
    p = argparse.ArgumentParser(description="Stage 6.5 batch: Grad-CAM++ + F-Fidelity")
    p.add_argument("--split", required=True)
    p.add_argument("--data-root", default=str(MSD_DIR))
    p.add_argument("--ckpt", required=True, help="stage 6.3 checkpoint")
    p.add_argument("--pred-masks", default="", help="predict_seg --save-masks dir (real path)")
    p.add_argument("--group", default="val")
    p.add_argument("--out", default=str(RESULTS_DIR / "final" / "stage65"))
    p.add_argument("--limit", type=int, default=0, help="first N cases only (quick check)")
    p.add_argument("--max-figures", type=int, default=4)
    p.add_argument("--stamp", action="store_true", help="phantom data: stamp the figures")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    model, cfg = load_cls_checkpoint(args.ckpt, args.device)
    split = rebase_split_to_msd(load_split(args.split), args.data_root)
    cases = split[args.group][: args.limit or None]
    mask_fn = pred_mask_fn(args.pred_masks) if args.pred_masks else None
    items = load_items(cfg, cases, args.group, tuple(cfg["roi_size"]), mask_fn=mask_fn)
    print(f"[6.5] {len(items)} patches from {len(cases)} {args.group} cases "
          f"({'predicted' if mask_fn else 'ground-truth'} masks)")

    df, cams = explain_items(model, items, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stage65_fidelity.csv", index=False)
    summary = {"stage": "6.5", "group": args.group, "checkpoint": str(args.ckpt),
               "masks": "predicted" if mask_fn else "ground truth", **summarize(df)}
    (out / "stage65_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    by_case = {it["case"]: it for it in items}
    top = df[df["y_true"] == 1].sort_values("prob_tumor", ascending=False).head(args.max_figures)
    for name in top["case"]:
        safe = name.replace("#", "_")
        overlay_saliency(by_case[name]["x"][None], cams[name], path=out / f"fig_saliency_{safe}.png",
                         title=f"{name}  p(tumor)={float(df.loc[df.case == name, 'prob_tumor'].iloc[0]):.2f}",
                         stamp=args.stamp)

    for key in ("overall", "tumor_patches", "tissue_patches", "sub2cm_tumor_patches"):
        r = summary[key]
        print(f"[6.5] {key:22s} n={r['n']:4d}  fidelity pass rate={r['pass_rate']}")
    print(f"[6.5] wrote {out}")
    return summary


if __name__ == "__main__":
    main()
