"""Stage 6.5 batch runner on the phantom: one row per patch, pass rates, figures."""
import json

import pandas as pd

from src.run_stage65 import main as stage65_main
from src.train_cls import main as train_main


def test_stage65_batch_writes_rows_summary_and_figures(msd_phantom, tmp_path):
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    train_main(["--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
                "--model", "resnet10", "--roi-size", "16", "--epochs", "1", "--batch", "2",
                "--device", "cpu", "--sampling", "patch", "--neg-per-case", "1"])
    split = next(out.glob("split_*cases.json"))
    summary = stage65_main(["--split", str(split), "--data-root", str(msd_phantom),
                            "--ckpt", str(ckpt / "stage63" / "best.pt"), "--group", "train",
                            "--out", str(out / "s65"), "--max-figures", "1", "--device", "cpu",
                            "--stamp"])
    df = pd.read_csv(out / "s65" / "stage65_fidelity.csv")
    assert len(df) == summary["overall"]["n"] > 0
    assert set(df["y_true"]) == {0, 1}
    assert df["fidelity_pass"].isin([True, False]).all()
    assert summary["tumor_patches"]["n"] + summary["tissue_patches"]["n"] == len(df)
    on_disk = json.loads((out / "s65" / "stage65_summary.json").read_text())
    assert on_disk["masks"] == "ground truth"
    assert len(list((out / "s65").glob("fig_saliency_*.png"))) == 1
