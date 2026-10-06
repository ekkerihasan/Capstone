"""The 6.1 -> 6.4 demo pipeline, end to end on phantoms.

Stage 6.2 is skipped here: it is covered by test_train_seg.py and training even a
tiny network twice would dominate the suite's runtime. What this file checks is
the wiring - that each stage hands the next one what it expects, and that the
manifest tells the truth about which stages were real.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.phantom import DEMO_STAMP
from src.run_demo_pipeline import main, placeholder_predictions

CASES = 40          # train/val/calib = 20/10/10; split conformal at alpha=0.1
                    # needs ceil((n+1)(1-alpha)) <= n, so at least 9 calibration cases


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_out")
    data = tmp_path_factory.mktemp("demo_data")
    manifest = main(["--out", str(out), "--data-root", str(data),
                     "--cases", str(CASES), "--skip-seg"])
    return out, manifest


def test_every_stage_is_recorded(demo):
    _, manifest = demo
    assert set(manifest["stages"]) == {"6.1", "6.2", "6.3", "6.4"}
    assert manifest["data"] == "SYNTHETIC PHANTOM"
    assert manifest["stamp"] == DEMO_STAMP


def test_manifest_admits_the_placeholder(demo):
    """The manifest is the only thing standing between a placeholder and a
    slide, so it has to say so in as many words."""
    _, manifest = demo
    assert manifest["stages"]["6.3"]["status"] == "PLACEHOLDER"
    assert manifest["stages"]["6.2"]["status"] == "skipped"
    assert manifest["stages"]["6.1"]["status"] == "ran"


def test_stage61_artifacts_exist(demo):
    out, _ = demo
    assert (out / "raw_stats.csv").exists()
    assert (out / "stage61_summary.json").exists()
    assert (out / "fig_size_distribution.png").exists()
    assert len(list(out.glob("split_*cases.json"))) == 1
    assert len(list(out.glob("overlay_*.png"))) > 0


def test_stage61_marks_its_figures_as_phantom(demo):
    out, _ = demo
    summary = json.loads((out / "stage61_summary.json").read_text(encoding="utf-8"))
    assert summary["stamped_phantom"] is True


def test_stage64_artifacts_exist(demo):
    out, _ = demo
    for name in ("stage64_summary.json", "reliability_bins.csv", "coverage_curve.csv",
                 "conformal_sets.csv", "stage64_by_size_bucket.csv",
                 "fig_stage64_reliability.png", "fig_stage64_conformal.png"):
        assert (out / name).exists(), name


def test_predictions_honour_the_63_to_64_contract(demo):
    out, _ = demo
    df = pd.read_csv(out / "cls_predictions.csv")
    assert list(df.columns) == ["case", "group", "y_true", "prob_tumor", "eq_diam_cm"]
    assert len(df) == CASES
    assert set(df["group"]) == {"train", "val", "calib"}
    assert df["prob_tumor"].between(0.0, 1.0).all()
    assert set(df["y_true"]) == {0, 1}


def test_both_size_buckets_are_populated(demo):
    """A demo whose >= 2 cm row is empty cannot show the sub-2 cm comparison the
    project is about, which is why the phantom pancreas is enlarged."""
    out, _ = demo
    df = pd.read_csv(out / "cls_predictions.csv")
    diam = df.loc[df.y_true == 1, "eq_diam_cm"]
    assert (diam < 2.0).any()
    assert (diam >= 2.0).any()


def test_placeholder_is_deterministic_and_separable():
    stats = pd.DataFrame({
        "case": [f"c{i:02d}" for i in range(20)],
        "split": ["val"] * 20,
        "has_tumor": [i % 2 for i in range(20)],
        "eq_diam_cm": [1.5 if i % 2 else 0.0 for i in range(20)],
    })
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.csv"
        b = Path(d) / "b.csv"
        placeholder_predictions(stats, a, seed=42)
        placeholder_predictions(stats, b, seed=42)
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")

        df = pd.read_csv(a)
        # generated, not predicted - but it must at least rank tumors above
        # non-tumors, or stage 6.4 would be calibrating noise
        assert df.loc[df.y_true == 1, "prob_tumor"].mean() > \
               df.loc[df.y_true == 0, "prob_tumor"].mean()


def test_placeholder_says_so_in_its_summary():
    stats = pd.DataFrame({"case": ["c0"], "split": ["val"], "has_tumor": [1],
                          "eq_diam_cm": [1.0]})
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        info = placeholder_predictions(stats, Path(d) / "p.csv", seed=0)
    assert info["status"] == "PLACEHOLDER"
    assert "not predicted" in info["note"]


def test_missing_predictions_file_is_fatal(tmp_path):
    """Pointing --predictions at nothing must stop, not silently fall back to
    the placeholder - that is how a placeholder reaches a slide."""
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "o"), "--data-root", str(tmp_path / "d"),
              "--cases", "6", "--skip-seg",
              "--predictions", str(tmp_path / "does_not_exist.csv")])
