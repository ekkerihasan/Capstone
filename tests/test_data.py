"""Stage 6.1 — dataset discovery and patient-wise splitting."""
import numpy as np
import pytest

from src.config import SEED
from src.data import (assert_disjoint, case_id, load_msd_datalist, load_split,
                      patient_wise_split, save_split)


def test_datalist_skips_macos_resource_forks(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    assert len(cases) == 6
    assert all(not case_id(c["image"]).startswith("._") for c in cases)
    assert all("image" in c and "label" in c for c in cases)


def test_datalist_is_sorted_and_absolute(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    ids = [case_id(c["image"]) for c in cases]
    assert ids == sorted(ids)
    assert all(str(msd_phantom) in c["image"] for c in cases)


def test_split_is_patient_disjoint_and_complete(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    split = patient_wise_split(cases, fractions=(0.5, 0.25, 0.25))
    assert assert_disjoint(split)
    assert sum(len(v) for v in split.values()) == len(cases)


def test_split_is_deterministic(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    a = patient_wise_split(cases, seed=SEED)
    b = patient_wise_split(list(reversed(cases)), seed=SEED)  # input order must not matter
    assert {k: [case_id(c["image"]) for c in v] for k, v in a.items()} == \
           {k: [case_id(c["image"]) for c in v] for k, v in b.items()}
    c = patient_wise_split(cases, seed=SEED + 1)
    assert [case_id(x["image"]) for x in a["train"]] != [case_id(x["image"]) for x in c["train"]]


def test_split_rejects_bad_fractions(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    with pytest.raises(ValueError):
        patient_wise_split(cases, fractions=(0.5, 0.4))


def test_split_roundtrips_through_json(msd_phantom, tmp_path):
    cases = load_msd_datalist(msd_phantom)
    split = patient_wise_split(cases)
    path = save_split(split, tmp_path / "split.json")
    assert load_split(path) == split


def test_assert_disjoint_catches_leakage(msd_phantom):
    cases = load_msd_datalist(msd_phantom)
    leaky = {"train": cases[:4], "val": cases[3:]}   # pancreas_003 in both
    with pytest.raises(AssertionError, match="pancreas_003"):
        assert_disjoint(leaky)
