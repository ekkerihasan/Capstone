"""Stage 6.3 classifier, ROI loading, training loop and the 6.3 -> 6.4 CSV contract."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.modules.dropout import _DropoutNd

from src.classifier import AVAILABLE, build_classifier, tumor_probability
from src.predict_cls import COLUMNS, main as predict_main
from src.train_cls import class_weights, evaluate, load_roi_cases, main as train_main, stack
from src.uncertainty import enable_dropout

ROI = (16, 16, 16)


@pytest.mark.parametrize("name", AVAILABLE)
def test_every_architecture_maps_a_roi_to_two_logits(name):
    model = build_classifier(name)
    out = model(torch.zeros(2, 1, 32, 32, 32))     # 32 is the DenseNet minimum (5 stride-2 stages)
    assert out.shape == (2, 2)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        build_classifier("vgg16")


@pytest.mark.parametrize("name", ["densenet121", "resnet10"])
def test_dropout_reaches_the_network_for_mc_dropout(name):
    model = build_classifier(name, dropout=0.3)
    assert len(enable_dropout(model)) >= 1, "MC Dropout would be a no-op on this architecture"
    assert all(m.p == 0.3 for m in model.modules() if isinstance(m, _DropoutNd))


def test_tumor_probability_is_a_probability_per_case():
    p = tumor_probability(build_classifier("resnet10"), torch.zeros(3, 1, *ROI))
    assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all()


def test_class_weights_balance_and_single_class_warning(capsys):
    w = class_weights(torch.tensor([0, 0, 0, 1]), "cpu")
    assert w[1] > w[0]
    assert class_weights(torch.tensor([1, 1, 1]), "cpu") is None
    assert "single class" in capsys.readouterr().out


def test_evaluate_gives_nan_auroc_for_one_class():
    model = build_classifier("resnet10")
    x = torch.zeros(3, 1, *ROI)
    res = evaluate(model, x, torch.tensor([1, 1, 1]), "cpu")
    assert np.isnan(res["auroc"]) and res["loss"] > 0


@pytest.fixture(scope="module")
def cases(msd_phantom):
    from src.data import load_msd_datalist
    return load_msd_datalist(msd_phantom)


def test_load_roi_cases_labels_and_shapes(cases):
    items = load_roi_cases(cases, "train", ROI)
    assert len(items) == len(cases) == 6
    assert all(it["x"].shape == (1, *ROI) and it["x"].dtype == np.float32 for it in items)
    # phantom: even-indexed cases carry a tumor, odd ones do not
    by_case = {it["case"]: it for it in items}
    assert by_case["pancreas_000"]["y_true"] == 1 and by_case["pancreas_001"]["y_true"] == 0
    assert by_case["pancreas_000"]["eq_diam_cm"] > 0 and by_case["pancreas_001"]["eq_diam_cm"] == 0
    assert {it["group"] for it in items} == {"train"}
    x, y = stack(items)
    assert x.shape == (6, 1, *ROI) and int(y.sum()) == 3


def test_case_without_pancreas_is_skipped_not_cropped(cases, capsys):
    items = load_roi_cases(cases[:2], "train", ROI, mask_fn=lambda c, label: np.zeros_like(label))
    assert items == []
    assert "skip" in capsys.readouterr().out


def test_train_then_predict_writes_the_stage64_contract(msd_phantom, tmp_path, capsys):
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    common = ["--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
              "--model", "resnet10", "--roi-size", "16", "--dropout", "0.2"]
    train_main(common + ["--epochs", "2", "--batch", "2", "--device", "cpu"])

    assert (ckpt / "stage63" / "last.pt").exists() and (ckpt / "stage63" / "best.pt").exists()
    history = json.loads((out / "cls_training_history.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1]
    split = next(out.glob("split_*cases.json"))

    # resume continues from epoch 2 rather than restarting
    train_main(common + ["--epochs", "3", "--batch", "2", "--device", "cpu", "--resume",
                         "--split", str(split)])
    assert [h["epoch"] for h in json.loads((out / "cls_training_history.json").read_text())] == [0, 1, 2]

    predict_main(["--split", str(split), "--data-root", str(msd_phantom), "--out", str(out),
                  "--ckpt", str(ckpt / "stage63" / "best.pt"), "--device", "cpu"])
    df = pd.read_csv(out / "cls_predictions.csv")
    assert list(df.columns) == COLUMNS                      # exact contract with stage 6.4
    assert len(df) == 6 and set(df["group"]) == {"train", "val", "calib"}
    assert df["prob_tumor"].between(0, 1).all() and set(df["y_true"]) <= {0, 1}

    predict_main(["--split", str(split), "--data-root", str(msd_phantom), "--out", str(out),
                  "--ckpt", str(ckpt / "stage63" / "best.pt"), "--device", "cpu", "--mc", "3"])
    assert {"mc_std", "entropy"} <= set(pd.read_csv(out / "cls_predictions.csv").columns)


def test_mc_flag_without_dropout_checkpoint_fails_loudly(msd_phantom, tmp_path):
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    train_main(["--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
                "--model", "resnet10", "--roi-size", "16", "--epochs", "1", "--batch", "2",
                "--device", "cpu"])
    split = next(out.glob("split_*cases.json"))
    with pytest.raises(SystemExit, match="dropout"):
        predict_main(["--split", str(split), "--data-root", str(msd_phantom), "--out", str(out),
                      "--ckpt", str(ckpt / "stage63" / "best.pt"), "--device", "cpu", "--mc", "3"])
