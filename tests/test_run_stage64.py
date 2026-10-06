"""Stage 6.4 runner, driven by a synthetic cls_predictions.csv in the agreed
layout (case, group, y_true, prob_tumor, eq_diam_cm)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.run_stage64 import load_predictions, main


def _predictions(path, n_calib=120, n_test=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for group, n in (("calib", n_calib), ("val", n_test), ("train", 10)):
        y = rng.integers(0, 2, size=n)
        p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.25, size=n), 0.01, 0.99)
        diam = np.where(y == 1, rng.uniform(0.8, 4.0, size=n), 0.0)
        for i in range(n):
            rows.append({"case": f"{group}_{i:03d}", "group": group, "y_true": int(y[i]),
                         "prob_tumor": float(p[i]), "eq_diam_cm": round(float(diam[i]), 2),
                         "mc_std": float(abs(rng.normal(0, 0.05)))})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("stage64")
    csv = _predictions(out / "cls_predictions.csv")
    summary = main(["--predictions", str(csv), "--out", str(out), "--alpha", "0.1", "--stamp"])
    return summary, out


def test_runner_writes_every_deliverable(run):
    summary, out = run
    for name in ("stage64_summary.json", "reliability_bins.csv", "coverage_curve.csv",
                 "conformal_sets.csv", "stage64_by_size_bucket.csv",
                 "fig_stage64_reliability.png", "fig_stage64_conformal.png"):
        assert (out / name).exists(), name
    assert json.loads((out / "stage64_summary.json").read_text())["stage"] == "6.4"


def test_summary_reports_overall_and_sub2cm(run):
    summary, _ = run
    assert set(summary["auroc"]) == {"overall", "sub2cm", "ge2cm"}
    assert 0.5 < summary["auroc"]["overall"]["auroc"] <= 1.0
    assert "sub2cm" in summary["by_size_bucket"]
    assert summary["n_test"] == 160 and summary["n_calib"] == 120
    assert summary["stamped_phantom"] is True


def test_conformal_block_is_consistent(run):
    summary, out = run
    c = summary["conformal"]
    assert c["alpha"] == 0.1 and 0.0 < c["q_hat"] <= 1.0
    assert c["coverage"] >= 0.85                       # 90 % target, finite-sample slack
    sets = pd.read_csv(out / "conformal_sets.csv")
    assert len(sets) == summary["n_test"]
    assert sets["covered"].mean() == pytest.approx(c["coverage"])
    assert (sets["abstain"] == (sets["set_size"] != 1)).all()


def test_calibration_metrics_are_bounded(run):
    summary, out = run
    assert 0.0 <= summary["ece"] <= 1.0
    assert 0.0 <= summary["sensitivity"] <= 1.0 and 0.0 <= summary["specificity"] <= 1.0
    assert sum(summary["confusion_tn_fp_fn_tp"]) == summary["n_test"]
    bins = pd.read_csv(out / "reliability_bins.csv")
    assert len(bins) == summary["n_bins"]


def test_load_predictions_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"case": ["a"], "prob_tumor": [0.5]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_predictions(bad)


def test_runner_fails_loudly_without_a_calibration_group(tmp_path):
    csv = tmp_path / "p.csv"
    pd.DataFrame({"case": ["a", "b"], "group": ["val", "val"], "y_true": [0, 1],
                  "prob_tumor": [0.2, 0.8], "eq_diam_cm": [0, 1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="calib"):
        main(["--predictions", str(csv), "--out", str(tmp_path), "--no-figures"])
