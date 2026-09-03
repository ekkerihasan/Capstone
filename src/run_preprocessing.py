"""Stage 6.1 end-to-end, as a script.

    python -m src.run_preprocessing --data-root data/Task07_Pancreas \
                                    --out results/review1 --limit 20

Produces, in --out:
    split_<n>cases.json     patient-wise split, reused by every later stage
    raw_stats.csv           per-case volumes and equivalent tumor diameter
    fig_size_distribution.png
    overlay_<case>.png      preprocessed CT + mask, one per validation case
    stage61_summary.json    counts, sub-2 cm tally, transform settings

Runs on CPU. Nothing here needs a GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, never shown

import nibabel as nib
import numpy as np
import pandas as pd
from monai.utils import set_determinism

from .config import HU_MAX, HU_MIN, MSD_DIR, RESULTS_DIR, SEED, SPACING_MM
from .data import assert_disjoint, case_id, load_msd_datalist, patient_wise_split, save_split
from .metrics import tumor_stats
from .transforms import build_transforms
from .viz import plot_case, plot_size_distribution


def raw_case_stats(cases):
    """Per-case statistics read straight from the NIfTI headers — no resampling,
    so tumor sizes are the true acquired ones."""
    rows = []
    for c in cases:
        img = nib.load(c["image"])
        spacing = img.header.get_zooms()[:3]
        label = np.asarray(nib.load(c["label"]).dataobj)
        row = {"case": case_id(c["image"]),
               "shape": "x".join(str(s) for s in img.shape[:3]),
               "spacing_mm": "/".join(f"{s:.2f}" for s in spacing)}
        row.update(tumor_stats(label, spacing))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 6.1 preprocessing")
    p.add_argument("--data-root", default=str(MSD_DIR), help="Task07_Pancreas directory")
    p.add_argument("--out", default=str(RESULTS_DIR / "review1"))
    p.add_argument("--limit", type=int, default=0, help="use only the first N cases (0 = all)")
    p.add_argument("--fractions", default="0.7,0.15,0.15")
    p.add_argument("--names", default="train,val,calib")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    set_determinism(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    cases = load_msd_datalist(args.data_root)
    if args.limit:
        cases = cases[: args.limit]
    print(f"[6.1] {len(cases)} labelled cases from {args.data_root}")

    fractions = tuple(float(x) for x in args.fractions.split(","))
    names = tuple(args.names.split(","))
    split = patient_wise_split(cases, fractions=fractions, names=names, seed=args.seed)
    assert_disjoint(split)
    split_path = save_split(split, out / f"split_{len(cases)}cases.json")
    print("[6.1] split", {k: len(v) for k, v in split.items()}, "->", split_path.name)

    stats = raw_case_stats(cases)
    group = {case_id(c["image"]): name for name, v in split.items() for c in v}
    stats.insert(1, "split", stats["case"].map(group))
    stats.to_csv(out / "raw_stats.csv", index=False)
    n_tumor = int(stats["has_tumor"].sum())
    n_small = int(stats["sub2cm"].sum())
    print(f"[6.1] cases with tumor: {n_tumor}/{len(stats)} | sub-2 cm: {n_small}")

    val_key = "val" if "val" in split else names[-1]
    if not args.no_figures:
        plot_size_distribution(stats, path=out / "fig_size_distribution.png")
        val_tf = build_transforms(train=False)
        for c in split[val_key]:
            name = case_id(c["image"])
            d = val_tf(dict(c))
            plot_case(d["image"], d["label"], title=name, path=out / f"overlay_{name}.png")
            print(f"[6.1] preprocessed {name}: {tuple(d['image'].shape[1:])} @ {SPACING_MM} mm")

    summary = {
        "stage": "6.1",
        "data_root": str(args.data_root),
        "seed": args.seed,
        "n_cases": len(cases),
        "split_counts": {k: len(v) for k, v in split.items()},
        "cases_with_tumor": n_tumor,
        "cases_sub2cm": n_small,
        "median_tumor_diam_cm": float(stats.loc[stats.has_tumor, "eq_diam_cm"].median())
        if n_tumor else None,
        "preprocessing": {
            "spacing_mm": list(SPACING_MM),
            "orientation": "RAS",
            "hu_window": [HU_MIN, HU_MAX],
            "normalised_to": [0.0, 1.0],
            "augmentation": ["RandFlipd", "RandRotate90d", "Rand3DElasticd"],
        },
        "runtime_s": round(time.time() - started, 1),
    }
    (out / "stage61_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[6.1] done in {summary['runtime_s']}s -> {out}")
    return summary


if __name__ == "__main__":
    main()
