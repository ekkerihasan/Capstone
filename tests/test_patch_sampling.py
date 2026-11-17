"""Stage 6.3 patch sampling: tumor vs non-tumor pancreatic tissue (real MSD path).

MSD has a tumor in every case, so negatives are tumor-free windows of the same
pancreas. The property that matters most: a negative never contains a tumor voxel.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.predict_cls import COLUMNS, main as predict_main
from src.roi import crop_at, sample_patch_starts
from src.train_cls import load_patch_items, main as train_main

SIZE = (16, 16, 12)


def _organ(with_tumor=True, tumor_at=(20, 50, 30), r=3):
    """A long thin 'pancreas' (label 1) with a small tumor (label 2) at one end."""
    m = np.zeros((40, 80, 60), dtype=np.uint8)
    m[15:25, 10:70, 25:35] = 1
    if with_tumor:
        z, y, x = np.ogrid[:40, :80, :60]
        ball = (z - tumor_at[0]) ** 2 + (y - tumor_at[1]) ** 2 + (x - tumor_at[2]) ** 2 <= r * r
        m[ball] = 2
    return m


def _window(m, starts):
    return crop_at(m, starts, SIZE)


@pytest.mark.parametrize("seed", range(10))
def test_negatives_never_contain_tumor_and_positive_always_does(seed):
    m = _organ()
    picks = sample_patch_starts(m, SIZE, n_neg=4, rng=np.random.default_rng(seed),
                                spacing_mm=(1, 1, 1))
    pos = [s for s, y in picks if y == 1]
    neg = [s for s, y in picks if y == 0]
    assert len(pos) == 1 and len(neg) == 4
    assert (_window(m, pos[0]) == 2).any()
    for s in neg:
        w = _window(m, s)
        assert not (w == 2).any(), "a negative window touches the tumor"
        assert (w == 1).any(), "a negative window has no pancreas tissue"


def test_positive_is_jittered_not_always_centred():
    m = _organ()
    starts = {sample_patch_starts(m, SIZE, 0, np.random.default_rng(s), spacing_mm=(1, 1, 1))[0][0]
              for s in range(20)}
    assert len(starts) > 1


def test_tumor_filling_the_organ_yields_fewer_negatives_not_wrong_ones():
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    m[5:15, 5:15, 5:15] = 1
    m[6:14, 6:14, 6:14] = 2
    picks = sample_patch_starts(m, SIZE, n_neg=3, rng=np.random.default_rng(0))
    assert [y for _, y in picks] == [1]            # positive only, no fake negative


def test_tumor_free_case_gives_negatives_only():
    picks = sample_patch_starts(_organ(with_tumor=False), SIZE, 2, np.random.default_rng(0))
    assert [y for _, y in picks] == [0, 0]


def test_empty_mask_gives_nothing():
    assert sample_patch_starts(np.zeros((10, 10, 10)), SIZE, 2, np.random.default_rng(0)) == []


# --------------------------------------------------- loader and CLI on the phantom


@pytest.fixture(scope="module")
def cases(msd_phantom):
    from src.data import load_msd_datalist
    return load_msd_datalist(msd_phantom)


def test_patch_items_are_deterministic_and_both_classes(cases):
    a = load_patch_items(cases, "train", (16, 16, 16), n_neg=2)
    b = load_patch_items(list(reversed(cases)), "train", (16, 16, 16), n_neg=2)
    by_name = {it["case"]: it for it in b}
    assert {it["case"] for it in a} == set(by_name)
    for it in a:                                    # same patch whatever the case order
        np.testing.assert_array_equal(it["x"], by_name[it["case"]]["x"])
    ys = {it["y_true"] for it in a}
    assert ys == {0, 1}
    assert all(it["eq_diam_cm"] == 0.0 for it in a if it["y_true"] == 0)
    assert all(it["case"].startswith(it["patient"] + "#") for it in a)


def test_patch_mode_train_then_predict_keeps_the_stage64_contract(msd_phantom, tmp_path):
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    train_main(["--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
                "--model", "resnet10", "--roi-size", "16", "--epochs", "1", "--batch", "2",
                "--device", "cpu", "--sampling", "patch", "--neg-per-case", "2"])
    split = next(out.glob("split_*cases.json"))
    predict_main(["--split", str(split), "--data-root", str(msd_phantom), "--out", str(out),
                  "--ckpt", str(ckpt / "stage63" / "best.pt"), "--device", "cpu"])
    df = pd.read_csv(out / "cls_predictions.csv")
    assert list(df.columns) == COLUMNS
    assert df["case"].str.contains("#").all()                # rows are patches
    assert set(df["y_true"]) == {0, 1}
    patients = df["case"].str.split("#").str[0]
    # patient-wise split survives: no patient's patches in two groups
    assert (df.assign(p=patients).groupby("p")["group"].nunique() == 1).all()
    assert len(json.loads(split.read_text())) >= 1


def test_predicted_masks_dir_feeds_the_real_path(msd_phantom, tmp_path, cases):
    """predict_seg --save-masks -> predict_cls --pred-masks. Masks here are the
    ground truth saved as .npy, which must give the same rows as the oracle path."""
    from src.data import case_id
    from src.transforms import build_transforms

    out, ckpt, masks = tmp_path / "out", tmp_path / "ckpt", tmp_path / "masks"
    train_main(["--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
                "--model", "resnet10", "--roi-size", "16", "--epochs", "1", "--batch", "2",
                "--device", "cpu", "--sampling", "patch"])
    split = next(out.glob("split_*cases.json"))
    tf = build_transforms(train=False)
    masks.mkdir()
    for c in cases:
        np.save(masks / f"{case_id(c['image'])}.npy", np.asarray(tf(c)["label"])[0].astype(np.uint8))

    common = ["--split", str(split), "--data-root", str(msd_phantom),
              "--ckpt", str(ckpt / "stage63" / "best.pt"), "--device", "cpu"]
    predict_main(common + ["--out", str(out / "oracle")])
    predict_main(common + ["--out", str(out / "real"), "--pred-masks", str(masks)])
    a = pd.read_csv(out / "oracle" / "cls_predictions.csv")
    b = pd.read_csv(out / "real" / "cls_predictions.csv")
    pd.testing.assert_frame_equal(a, b)
