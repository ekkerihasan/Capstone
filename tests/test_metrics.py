"""Metrics: Dice, the sub-2 cm size bucket, and the stage 6.3/6.4 classification
and calibration metrics."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (auroc, auroc_by_size_bucket, by_size_bucket, confusion, dice,
                         equivalent_spherical_diameter_cm, expected_calibration_error,
                         reliability_curve, sensitivity_specificity, size_bucket,
                         tumor_stats, voxel_volume_ml)


def test_dice_bounds():
    a = np.zeros((8, 8, 8), bool); a[2:6, 2:6, 2:6] = True
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0   # both empty = agreement


def test_dice_half_overlap():
    a = np.zeros(100, bool); a[:50] = True
    b = np.zeros(100, bool); b[25:75] = True
    assert dice(a, b) == 0.5


def test_voxel_volume_ml():
    assert voxel_volume_ml((1.5, 1.5, 2.0)) == 4.5 / 1000.0
    assert voxel_volume_ml((10.0, 10.0, 10.0)) == 1.0        # 1 cm^3 = 1 mL


def test_equivalent_diameter_recovers_a_known_sphere():
    r_cm = 0.75                                              # 1.5 cm tumor
    volume_ml = 4.0 / 3.0 * np.pi * r_cm ** 3
    assert equivalent_spherical_diameter_cm(volume_ml) == np.float32(1.5).item() or \
           abs(equivalent_spherical_diameter_cm(volume_ml) - 1.5) < 1e-9
    assert equivalent_spherical_diameter_cm(0.0) == 0.0


def test_tumor_stats_flags_sub2cm():
    label = np.zeros((40, 40, 40), np.uint8)
    label[10:30, 10:30, 10:30] = 1
    label[18:22, 18:22, 18:22] = 2                           # 64 voxels
    stats = tumor_stats(label, spacing_mm=(1.5, 1.5, 2.0))
    assert stats["tumor_vox"] == 64
    assert stats["has_tumor"] and stats["sub2cm"]
    assert 0 < stats["eq_diam_cm"] < 2.0


def test_tumor_stats_on_a_tumor_free_case():
    label = np.zeros((20, 20, 20), np.uint8); label[5:15, 5:15, 5:15] = 1
    stats = tumor_stats(label, spacing_mm=(1.5, 1.5, 2.0))
    assert not stats["has_tumor"] and not stats["sub2cm"]
    assert stats["eq_diam_cm"] == 0.0


def test_size_bucket_edges():
    assert size_bucket(0.0) == "no_tumor"
    assert size_bucket(1.99) == "sub2cm"
    assert size_bucket(2.0) == "ge2cm"


def test_by_size_bucket_reports_overall_and_strata():
    df = pd.DataFrame({"eq_diam_cm": [0.0, 1.0, 1.5, 3.0], "dice_tumor": [1.0, 0.2, 0.4, 0.8]})
    table = by_size_bucket(df, ["dice_tumor"])
    assert table.loc["overall", "dice_tumor"] == 0.6
    assert table.loc["sub2cm", "dice_tumor"] == 0.3
    assert table.loc["sub2cm", "n"] == 2
    assert table.loc["ge2cm", "n"] == 1


# ------------------------------------------------------ stage 6.3 / 6.4 additions


def test_auroc_perfect_reversed_and_undefined():
    y = [0, 0, 1, 1]
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1, 1], [0.2, 0.5, 0.9]))      # one class -> undefined


def test_confusion_layout_is_tn_fp_fn_tp():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 1]
    assert confusion(y_true, y_pred).tolist() == [[2, 1], [1, 3]]


def test_sensitivity_specificity():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 1]
    sens, spec = sensitivity_specificity(y_true, y_pred)
    assert sens == pytest.approx(3 / 4) and spec == pytest.approx(2 / 3)
    sens, spec = sensitivity_specificity([0, 0], [0, 1])
    assert np.isnan(sens) and spec == 0.5                    # no positives -> NaN sensitivity


def test_ece_is_zero_when_every_bin_is_calibrated():
    # bin [0.2, 0.3): p=0.25 with 1 in 4 positive; bin [0.7, 0.8): p=0.75 with 3 in 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    p = [0.25] * 4 + [0.75] * 4
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_equals_the_gap_for_one_overconfident_bin():
    y = [1, 0, 0, 0]                 # 25 % positive
    p = [0.95] * 4                   # the model says 95 %
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.70)


def test_ece_weights_bins_by_count():
    y = [1, 0] + [0] * 8             # bin A: 2 cases, 50 % pos; bin B: 8 cases, 0 % pos
    p = [0.55, 0.55] + [0.15] * 8
    expected = 0.2 * abs(0.5 - 0.55) + 0.8 * abs(0.0 - 0.15)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(expected)


def test_reliability_curve_shapes_and_empty_bins_are_nan():
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    p = [0.25] * 4 + [0.75] * 4
    centre, acc, conf = reliability_curve(y, p, n_bins=10)
    assert centre.shape == acc.shape == conf.shape == (10,)
    assert centre[2] == pytest.approx(0.25) and acc[2] == pytest.approx(0.25)
    assert conf[7] == pytest.approx(0.75) and acc[7] == pytest.approx(0.75)
    assert np.isnan(acc[0]) and np.isnan(conf[9])            # nothing landed there


def test_reliability_curve_keeps_p_equal_to_one_in_the_last_bin():
    centre, acc, conf = reliability_curve([1, 1], [1.0, 1.0], n_bins=4)
    assert acc[-1] == 1.0 and conf[-1] == 1.0 and np.isnan(acc[0])


def test_auroc_by_size_bucket_ranks_small_tumors_against_all_negatives():
    df = pd.DataFrame({
        "y_true":     [0,   0,   0,   1,   1,   1,   1],
        "prob_tumor": [0.1, 0.3, 0.5, 0.2, 0.4, 0.9, 0.95],
        "eq_diam_cm": [0.0, 0.0, 0.0, 1.0, 1.5, 3.0, 4.0],   # two sub-2 cm, two >= 2 cm
    })
    table = auroc_by_size_bucket(df)
    assert list(table.index) == ["overall", "sub2cm", "ge2cm"]
    assert table.loc["ge2cm", "auroc"] == 1.0                 # big tumors rank above every negative
    assert table.loc["sub2cm", "auroc"] < 1.0                 # small ones do not
    assert table.loc["sub2cm", "n_pos"] == 2 and table.loc["sub2cm", "n_neg"] == 3
    assert table.loc["overall", "auroc"] == pytest.approx(auroc(df["y_true"], df["prob_tumor"]))
