"""Stages 6.1 -> 6.4 end to end on synthetic phantoms, in one command.

    python -m src.run_demo_pipeline --out results/review2_demo

The lab RTX 4060 is unavailable, so no stage can be run on real data yet. This
runner exists so the *pipeline* is demonstrably runnable in the meantime: it
builds a phantom dataset, preprocesses it (6.1), optionally trains the Attention
U-Net for a couple of epochs (6.2), produces per-case tumor probabilities (6.3)
and calibrates them (6.4), writing every artifact the review deck needs.

Every figure it produces is stamped SYNTHETIC PHANTOM - PIPELINE DEMONSTRATION,
NOT A RESULT. Nothing here is a result. The numbers describe a phantom whose
answers we generated ourselves; they say the code runs, nothing more.

Stage 6.3 belongs to another lane and may not exist yet. Until it does, this
runner writes a clearly-flagged *placeholder* probability per case so the 6.4
half of the chain can run. Once stage 6.3 produces a real cls_predictions.csv,
pass it with --predictions and the placeholder is not used at all:

    python -m src.run_demo_pipeline --predictions <path to cls_predictions.csv>

manifest.json records, for every stage, whether it ran for real or was a
placeholder. Read it before putting any of this in a slide.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .config import DATA_DIR, RESULTS_DIR, SEED
from .phantom import DEMO_STAMP, write_phantom_dataset

# A pancreas large enough that a tumor inside it can exceed 2 cm equivalent
# diameter. The default phantom pancreas is ~1.6 cm across, so every tumor in it
# is sub-2 cm and the size-bucket table would show an empty >= 2 cm row.
DEMO_PANCREAS_AXES = (14.0, 9.0, 11.0)
DEMO_TUMOR_RADII = (3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)   # ~0.7 cm to ~2.3 cm


def build_phantom_data(root: Path, n_cases: int) -> Path:
    print(f"[demo] building {n_cases} phantom cases -> {root}")
    return write_phantom_dataset(root, n_cases=n_cases, include_junk=True,
                                 tumor_radii=DEMO_TUMOR_RADII,
                                 pancreas_axes=DEMO_PANCREAS_AXES)


def run_stage61(data_root: Path, out: Path, seed: int) -> dict:
    from . import run_preprocessing
    return run_preprocessing.main([
        "--data-root", str(data_root), "--out", str(out),
        "--fractions", "0.5,0.25,0.25", "--names", "train,val,calib",
        "--seed", str(seed), "--stamp",
    ])


def run_stage62(data_root: Path, out: Path, split_path: Path, seed: int,
                epochs: int) -> dict:
    """A deliberately tiny training run. The point is that the loop, the
    checkpointing and the resume path execute, not that the Dice means
    anything - on a phantom it does not."""
    from . import train_seg
    return train_seg.main([
        "--data-root", str(data_root), "--split", str(split_path),
        "--out", str(out), "--ckpt-root", str(out / "checkpoints"),
        "--epochs", str(epochs), "--batch", "1", "--patch", "32",
        "--cache-rate", "0.0", "--workers", "0", "--seed", str(seed),
        "--device", "cpu", "--no-amp",
    ])


def placeholder_predictions(stats: pd.DataFrame, out_path: Path, seed: int) -> dict:
    """Stand in for stage 6.3 until that lane lands.

    The probabilities are generated, not predicted. Larger tumors get a stronger
    signal and sub-2 cm ones sit near the decision boundary, which is the shape a
    real detector has, and the scores are deliberately over-confident so the
    calibration figures in 6.4 have something to show. None of it is a
    measurement of anything.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in stats.itertuples():
        y = int(r.has_tumor)
        signal = 0.2 + 1.6 * min(float(r.eq_diam_cm), 3.0) / 3.0 if y else -0.8
        prob = 1.0 / (1.0 + np.exp(-(signal + rng.normal(0.0, 0.9))))
        rows.append({
            "case": r.case,
            "group": r.split,
            "y_true": y,
            "prob_tumor": round(float(np.clip(prob ** 0.75, 0.01, 0.99)), 4),
            "eq_diam_cm": float(r.eq_diam_cm),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[6.3] PLACEHOLDER probabilities for {len(df)} cases -> {out_path.name}")
    return {"stage": "6.3", "status": "PLACEHOLDER", "n": len(df),
            "note": "generated, not predicted; replace with the real "
                    "cls_predictions.csv via --predictions"}


def run_stage64(predictions: Path, out: Path, alpha: float) -> dict:
    from . import run_stage64 as s64
    return s64.main([
        "--predictions", str(predictions), "--out", str(out),
        "--alpha", str(alpha), "--calib-group", "calib", "--test-group", "val",
        "--stamp",
    ]) or {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(RESULTS_DIR / "review2_demo"))
    p.add_argument("--data-root", default=str(DATA_DIR / "phantom"),
                   help="where the phantom volumes are written (data/ is gitignored)")
    p.add_argument("--cases", type=int, default=80,
                   help="phantom cases; 80 gives train/val/calib = 40/20/20, enough to "
                        "populate both tumor-size buckets in the held-out group")
    p.add_argument("--predictions", default="",
                   help="a real cls_predictions.csv from stage 6.3; without it a "
                        "clearly-flagged placeholder is generated")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seg-epochs", type=int, default=2)
    p.add_argument("--skip-seg", action="store_true",
                   help="skip the stage 6.2 training run (it is the slow part)")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {"pipeline": "6.1 -> 6.4", "data": "SYNTHETIC PHANTOM",
                "stamp": DEMO_STAMP, "seed": args.seed, "stages": {}}

    data_root = build_phantom_data(Path(args.data_root), args.cases)

    manifest["stages"]["6.1"] = {"status": "ran", **run_stage61(data_root, out, args.seed)}
    split_path = next(out.glob("split_*cases.json"))
    stats = pd.read_csv(out / "raw_stats.csv")

    if args.skip_seg:
        manifest["stages"]["6.2"] = {"status": "skipped",
                                     "note": "run without --skip-seg, or train on the lab 4060"}
        print("[6.2] skipped")
    else:
        manifest["stages"]["6.2"] = {
            "status": "ran (phantom, not a convergence claim)",
            **run_stage62(data_root, out, split_path, args.seed, args.seg_epochs)}

    if args.predictions:
        preds = Path(args.predictions)
        if not preds.exists():
            sys.exit(f"--predictions {preds} does not exist")
        manifest["stages"]["6.3"] = {"stage": "6.3", "status": "real",
                                     "source": str(preds)}
        print(f"[6.3] using real predictions from {preds}")
    else:
        preds = out / "cls_predictions.csv"
        manifest["stages"]["6.3"] = placeholder_predictions(stats, preds, args.seed)

    manifest["stages"]["6.4"] = {"status": "ran", **run_stage64(preds, out, args.alpha)}

    manifest["runtime_s"] = round(time.time() - started, 1)
    manifest["artifacts"] = sorted(f.name for f in out.iterdir() if f.is_file())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    placeholder = manifest["stages"]["6.3"]["status"] == "PLACEHOLDER"
    print(f"\n[demo] {manifest['runtime_s']}s -> {out}")
    print(f"[demo] {len(manifest['artifacts'])} artifacts, manifest.json records what is real")
    if placeholder:
        print("[demo] stage 6.3 is a PLACEHOLDER - rerun with --predictions once that lane lands")
    return manifest


if __name__ == "__main__":
    main()
