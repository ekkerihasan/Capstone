"""Stage 6.4 end-to-end, post hoc on the stage 6.3 predictions.

    python -m src.run_stage64 --predictions results/review2_demo/cls_predictions.csv \
                              --out results/review2_demo --alpha 0.1 [--stamp]

Input is the one CSV that crosses the 6.3 -> 6.4 line
(docs/REVIEW2_TEMP_PLAN.md section 9):

    case, group, y_true, prob_tumor, eq_diam_cm        (+ optional mc_std, entropy)

`prob_tumor` is expected to be the MC-Dropout mean (uncertainty.mc_dropout_table);
a plain softmax works too, the calibration numbers just describe that instead.

Writes to --out:
    stage64_summary.json       AUROC / sensitivity / specificity / ECE on the test group,
                               q_hat, coverage, set sizes, abstention rate — overall AND
                               for the sub-2 cm bucket
    reliability_bins.csv       the reliability curve
    coverage_curve.csv         coverage vs target over an alpha sweep
    conformal_sets.csv         per-case prediction set, size, covered flag
    fig_stage64_reliability.png
    fig_stage64_conformal.png

--stamp marks every figure with phantom.DEMO_STAMP. Mandatory for phantom runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, SMALL_TUMOR_CM
from .conformal import (ALPHA, calibrate, coverage_curve, predict_set, set_sizes,
                        summarize_sets)
from .metrics import (auroc_by_size_bucket, by_size_bucket, confusion,
                      expected_calibration_error, reliability_curve,
                      sensitivity_specificity, size_bucket)

REQUIRED = ("case", "group", "y_true", "prob_tumor", "eq_diam_cm")
DPI = 150


def _stamp(fig, stamp: bool):
    if stamp:
        from .phantom import stamp_figure
        stamp_figure(fig)


def _save(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def load_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {list(REQUIRED)}")
    df["y_true"] = df["y_true"].astype(int)
    df["prob_tumor"] = df["prob_tumor"].astype(float).clip(0.0, 1.0)
    df["eq_diam_cm"] = df["eq_diam_cm"].astype(float).fillna(0.0)
    df["size_bucket"] = df["eq_diam_cm"].map(size_bucket)
    return df


def plot_reliability(test: pd.DataFrame, ece: float, path, n_bins: int, stamp: bool,
                     title_group: str):
    import matplotlib.pyplot as plt

    centre, acc, conf = reliability_curve(test["y_true"], test["prob_tumor"], n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(test["prob_tumor"], bins=edges)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ok = ~np.isnan(acc)
    ax[0].bar(centre[ok], acc[ok], width=1.0 / n_bins, color="#5b4fbf", alpha=0.85,
              edgecolor="white", label="observed tumor fraction")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    ax[0].plot(conf[ok], acc[ok], "o", color="#d1495b", ms=4, label="bin mean")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("predicted p(tumor)")
    ax[0].set_ylabel("observed fraction with tumor")
    ax[0].set_title(f"Reliability diagram ({title_group}, n={len(test)}) | ECE = {ece:.3f}")
    ax[0].legend(loc="upper left", fontsize=8)

    ax[1].bar(centre, counts, width=1.0 / n_bins, color="#9a93d6", edgecolor="white")
    ax[1].set_xlabel("predicted p(tumor)")
    ax[1].set_ylabel("cases")
    ax[1].set_title("Confidence histogram")
    fig.tight_layout()
    _stamp(fig, stamp)
    return _save(fig, path)


def plot_conformal(curve: pd.DataFrame, sets_df: pd.DataFrame, alpha: float, q_hat: float,
                   path, stamp: bool):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(curve["target_coverage"], curve["coverage"], "-o", ms=3, color="#5b4fbf",
               label="empirical (test group)")
    ax[0].plot([0.5, 1], [0.5, 1], "--", color="grey", lw=1, label="target 1 - alpha")
    ax[0].axvline(1 - alpha, color="#d1495b", ls=":", lw=1, label=f"alpha = {alpha:g}")
    ax[0].set_xlabel("target coverage 1 - alpha")
    ax[0].set_ylabel("empirical coverage")
    ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title("Split-conformal coverage")
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].plot(curve["target_coverage"], curve["mean_set_size"], "-o", ms=3, color="#d1495b")
    ax[1].set_xlabel("target coverage 1 - alpha")
    ax[1].set_ylabel("mean prediction-set size")
    ax[1].set_ylim(0, 2.1)
    ax[1].set_title("Price of coverage: set size")

    sizes = sets_df["set_size"]
    labels = ["empty (0)", "single (1)", "both (2)"]
    vals = [int((sizes == k).sum()) for k in (0, 1, 2)]
    ax[2].bar(labels, vals, color=["#bbbbbb", "#5b4fbf", "#d1495b"])
    for i, v in enumerate(vals):
        ax[2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax[2].set_ylabel("cases")
    ax[2].set_title(f"Prediction sets at alpha={alpha:g} (q_hat={q_hat:.3f})\n"
                    f"size != 1 -> abstain, flag for specialist")
    fig.tight_layout()
    _stamp(fig, stamp)
    return _save(fig, path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 6.4 calibration + conformal, post hoc")
    p.add_argument("--predictions", required=True, help="cls_predictions.csv from stage 6.3")
    p.add_argument("--out", default=str(RESULTS_DIR / "review2"))
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--calib-group", default="calib")
    p.add_argument("--test-group", default="val")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold for sens/spec")
    p.add_argument("--stamp", action="store_true", help="mark figures as phantom-derived")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    calib = df[df["group"] == args.calib_group].reset_index(drop=True)
    test = df[df["group"] == args.test_group].reset_index(drop=True)
    if calib.empty or test.empty:
        raise ValueError(f"need rows in group '{args.calib_group}' and '{args.test_group}'; "
                         f"found groups {sorted(df['group'].unique())}")
    print(f"[6.4] {args.predictions}: calib={len(calib)} test={len(test)} "
          f"(test positives={int(test['y_true'].sum())}, "
          f"sub-2 cm={int((test['size_bucket'] == 'sub2cm').sum())})")

    # ---- discrimination and calibration on the test group ---------------------
    y, prob = test["y_true"].to_numpy(), test["prob_tumor"].to_numpy()
    pred = (prob >= args.threshold).astype(int)
    sens, spec = sensitivity_specificity(y, pred)
    ece = expected_calibration_error(y, prob, args.n_bins)
    auc_table = auroc_by_size_bucket(test)
    centre, acc, conf = reliability_curve(y, prob, args.n_bins)
    pd.DataFrame({"bin_centre": centre, "accuracy": acc, "confidence": conf}).to_csv(
        out / "reliability_bins.csv", index=False)

    # ---- split conformal: calibrate on calib, evaluate on test -----------------
    q_hat = calibrate(calib["prob_tumor"], calib["y_true"], alpha=args.alpha)
    sets = predict_set(test["prob_tumor"], q_hat)
    sets_df = test[["case", "group", "y_true", "prob_tumor", "eq_diam_cm", "size_bucket"]].copy()
    sets_df["set"] = ["|".join(map(str, s)) for s in sets]
    sets_df["set_size"] = set_sizes(sets)
    sets_df["covered"] = [int(t) in s for s, t in zip(sets, test["y_true"])]
    sets_df["abstain"] = sets_df["set_size"] != 1
    sets_df.to_csv(out / "conformal_sets.csv", index=False)

    curve = pd.DataFrame(coverage_curve(calib["prob_tumor"], calib["y_true"],
                                        test["prob_tumor"], test["y_true"]))
    curve.to_csv(out / "coverage_curve.csv", index=False)

    # The CLAUDE.md rule: overall AND sub-2 cm. by_size_bucket is metric-agnostic.
    per_case = sets_df.assign(covered=sets_df["covered"].astype(float),
                              abstain=sets_df["abstain"].astype(float),
                              correct=(pred == y).astype(float))
    bucket_table = by_size_bucket(per_case, ["covered", "abstain", "set_size", "correct"])
    bucket_table.to_csv(out / "stage64_by_size_bucket.csv")

    summary = {
        "stage": "6.4",
        "predictions": str(args.predictions),
        "calib_group": args.calib_group, "n_calib": int(len(calib)),
        "test_group": args.test_group, "n_test": int(len(test)),
        "n_test_positive": int(y.sum()),
        "n_test_sub2cm": int((test["size_bucket"] == "sub2cm").sum()),
        "threshold": args.threshold,
        "auroc": json.loads(auc_table.to_json(orient="index")),
        "sensitivity": sens, "specificity": spec,
        "confusion_tn_fp_fn_tp": confusion(y, pred).ravel().tolist(),
        "ece": round(ece, 4), "n_bins": args.n_bins,
        "mc_std_mean": round(float(test["mc_std"].mean()), 4) if "mc_std" in test else None,
        "conformal": {"alpha": args.alpha, "target_coverage": 1 - args.alpha, "q_hat": round(q_hat, 4),
                      **summarize_sets(sets, test["y_true"])},
        "by_size_bucket": json.loads(bucket_table.to_json(orient="index")),
        "sub2cm_threshold_cm": SMALL_TUMOR_CM,
        "stamped_phantom": bool(args.stamp),
    }
    (out / "stage64_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if not args.no_figures:
        plot_reliability(test, ece, out / "fig_stage64_reliability.png", args.n_bins,
                         args.stamp, args.test_group)
        plot_conformal(curve, sets_df, args.alpha, q_hat, out / "fig_stage64_conformal.png",
                       args.stamp)

    print(auc_table.to_string())
    print(f"[6.4] sens={sens:.3f} spec={spec:.3f} ECE={ece:.3f} | "
          f"conformal alpha={args.alpha}: q_hat={q_hat:.3f} "
          f"coverage={summary['conformal']['coverage']:.3f} "
          f"abstain={summary['conformal']['frac_abstain']:.3f} -> {out}")
    return summary


if __name__ == "__main__":
    main()
