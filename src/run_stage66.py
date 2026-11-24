"""Stage 6.6 case-level selective-prediction evaluation on a split group.

    python -m src.run_stage66 --group val --device cuda

The classifier and conformal threshold are patch-based, while the app returns
one decision per patient by selecting the highest-scoring pancreas window.
This runner evaluates that same app path at the patient level.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from monai.utils import set_determinism

from .app import analyse, checkpoint_patch, load_classifier, load_model, training_patch
from .config import (CHECKPOINT_DIR, LABEL_TUMOR, MSD_DIR, RESULTS_DIR, SEED, SPACING_MM,
                     portable_path)
from .data import load_split, rebase_split_to_msd
from .metrics import equivalent_spherical_diameter_cm, size_bucket, voxel_volume_ml


def summarize(df: pd.DataFrame) -> dict:
    def summarize_group(group: pd.DataFrame) -> dict:
        accepted = group[~group["abstain"]]
        return {
            "n_cases": int(len(group)),
            "accepted": int(len(accepted)),
            "abstention_rate": round(float(group["abstain"].mean()), 4) if len(group) else None,
            "accepted_accuracy": (round(float(accepted["correct"].mean()), 4)
                                  if len(accepted) else None),
        }

    return {
        "overall": summarize_group(df),
        "sub2cm": summarize_group(df[df["size_bucket"] == "sub2cm"]),
        "ge2cm": summarize_group(df[df["size_bucket"] == "ge2cm"]),
        "no_tumor": summarize_group(df[df["size_bucket"] == "no_tumor"]),
    }


def build_argparser():
    p = argparse.ArgumentParser(description="Stage 6.6 case-level selective-prediction evaluation")
    p.add_argument("--split", default=str(RESULTS_DIR / "review1_local" / "full281" /
                                           "split_281cases.json"))
    p.add_argument("--data-root", default=str(MSD_DIR))
    p.add_argument("--group", default="val")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "stage62" / "best.pt"))
    p.add_argument("--classifier", default=str(CHECKPOINT_DIR / "stage63" / "best.pt"))
    p.add_argument("--calibration", default=str(RESULTS_DIR / "final" / "real" /
                                                 "stage64_summary.json"))
    p.add_argument("--out", default=str(RESULTS_DIR / "final" / "stage66"))
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="first N cases only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_determinism(SEED)
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    q_hat = calibration["conformal"]["q_hat"]
    alpha = calibration["conformal"]["alpha"]

    device = torch.device(args.device)
    segmentation, trained = load_model(Path(args.checkpoint), device=device)
    classifier, cfg = load_classifier(Path(args.classifier), device=device)
    patch = checkpoint_patch(Path(args.checkpoint)) or training_patch()
    split = rebase_split_to_msd(load_split(args.split), args.data_root)
    cases = split[args.group][:args.limit or None]

    rows = []
    for item in cases:
        path = Path(item["image"])
        fig, verdict, gt, _, _ = analyse(
            path, segmentation, trained, q_hat, alpha, mc_passes=args.mc_passes,
            patch=patch, classifier=classifier, cls_cfg=cfg, want_saliency=True,
            phantom=False,
        )
        if gt is None:
            raise ValueError(f"{path}: expected a ground-truth label for evaluation")
        y_true = int((gt == LABEL_TUMOR).any())
        volume_ml = int((gt == LABEL_TUMOR).sum()) * voxel_volume_ml(SPACING_MM)
        diameter = equivalent_spherical_diameter_cm(volume_ml) if y_true else 0.0
        abstain = bool(verdict["abstain"])
        decision = verdict["decision"]
        pred = 1 if decision == "tumor" else 0
        fidelity = verdict.get("fidelity")
        rows.append({
            "case": path.name.replace(".nii.gz", ""),
            "y_true": y_true,
            "decision": decision,
            "abstain": abstain,
            "correct": None if abstain else int(pred == y_true),
            "prob_tumor": verdict["prob_tumor"],
            "mc_std": verdict["mc_std"],
            "conformal_set": "|".join(map(str, verdict["conformal_set"] or [])),
            "fidelity_pass": None if fidelity is None else bool(fidelity["passed"]),
            "eq_diam_cm": diameter,
            "size_bucket": size_bucket(diameter),
            "n_windows": verdict.get("n_windows", 0),
        })
        import matplotlib.pyplot as plt
        plt.close(fig)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stage66_per_case.csv", index=False)
    summary = {
        "stage": "6.6",
        "unit": "patient/case",
        "group": args.group,
        "n_cases": int(len(df)),
        "checkpoint": portable_path(args.checkpoint),
        "classifier": portable_path(args.classifier),
        "calibration": portable_path(calibration_path),
        "alpha": alpha,
        "q_hat": q_hat,
        "mc_passes": args.mc_passes,
        "device": str(device),
        "stamped_phantom": False,
        **summarize(df),
    }
    (out / "stage66_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
