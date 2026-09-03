"""Metrics: Dice and the sub-2 cm size bucket."""
import numpy as np
import pandas as pd

from src.metrics import (by_size_bucket, dice, equivalent_spherical_diameter_cm,
                         size_bucket, tumor_stats, voxel_volume_ml)


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
