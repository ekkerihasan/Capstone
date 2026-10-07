"""Stage 6.5 - the F-Fidelity masking test.

Two tests carry this file: a faithful saliency map must pass, and a map that
points at nothing in particular must fail. A test that only ever sees one of
those cannot tell a working check from a check that always says the same thing.
"""
import numpy as np
import pytest
import torch

from src.explain import gradcam_plusplus
from src.fidelity import (MIN_MARGIN, fidelity_test, mask_random, mask_top_k,
                          plot_fidelity)
from tests.test_explain import CornerDetector


@pytest.fixture
def corner_case():
    x = torch.zeros(1, 1, 16, 16, 16)
    x[0, 0, :8, :8, :8] = 1.0
    return CornerDetector().eval(), x


def test_mask_top_k_blanks_the_most_salient_voxels():
    x = torch.ones(1, 1, 4, 4, 4)
    sal = np.zeros((4, 4, 4))
    sal[0, 0, 0] = 1.0
    out = mask_top_k(x, sal, k_frac=1 / 64)
    assert out[0, 0, 0, 0, 0] == 0.0
    assert out.sum() == 63.0


def test_mask_top_k_rounds_up_so_a_tiny_fraction_still_masks_something():
    x = torch.ones(1, 1, 4, 4, 4)
    out = mask_top_k(x, np.random.default_rng(0).random((4, 4, 4)), k_frac=1e-6)
    assert out.sum() == 63.0


def test_mask_random_blanks_the_requested_fraction():
    x = torch.ones(1, 1, 10, 10, 10)
    out = mask_random(x, k_frac=0.1, rng=np.random.default_rng(0))
    assert out.sum() == pytest.approx(900.0)


def test_masking_does_not_touch_the_original():
    x = torch.ones(1, 1, 4, 4, 4)
    mask_top_k(x, np.ones((4, 4, 4)), k_frac=0.5)
    mask_random(x, k_frac=0.5, rng=np.random.default_rng(0))
    assert x.sum() == 64.0


def test_a_faithful_map_passes(corner_case):
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    res = fidelity_test(model, x, cam, class_idx=cls,
                        k_fractions=(0.05, 0.10, 0.20), n_random=5)
    assert res["passed"] is True
    assert res["mean_margin"] > MIN_MARGIN


def test_a_meaningless_map_fails(corner_case):
    """Same model, saliency pointing away from the evidence. Masking there
    should do no better than masking at random."""
    model, x = corner_case
    sal = np.zeros((16, 16, 16))
    sal[8:, 8:, 8:] = 1.0                      # the opposite corner
    res = fidelity_test(model, x, sal, class_idx=1,
                        k_fractions=(0.05, 0.10, 0.20), n_random=5)
    assert res["passed"] is False


def test_uniform_saliency_fails():
    """A map that highlights everything explains nothing, and must not pass by
    virtue of masking a lot."""
    model = CornerDetector().eval()
    x = torch.zeros(1, 1, 16, 16, 16)
    x[0, 0, :8, :8, :8] = 1.0
    res = fidelity_test(model, x, np.ones((16, 16, 16)), class_idx=1,
                        k_fractions=(0.05, 0.10), n_random=5)
    assert res["passed"] is False


def test_zero_margin_cannot_pass_on_zero_variance(corner_case):
    """With a saturated model both drops can be exactly 0 and the random std 0
    too. Without the absolute floor, `margin > z * std` would read 0 > 0 as a
    pass on some platforms; the floor makes it unambiguous."""
    model, x = corner_case
    res = fidelity_test(model, x, np.zeros((16, 16, 16)), class_idx=1,
                        k_fractions=(0.001,), n_random=3)
    assert res["threshold"] >= MIN_MARGIN
    assert res["passed"] is False


def test_result_reports_the_whole_curve(corner_case):
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    res = fidelity_test(model, x, cam, class_idx=cls, k_fractions=(0.05, 0.20), n_random=3)
    assert res["n_k"] == 2 and len(res["curve"]) == 2
    for row in res["curve"]:
        assert {"k_frac", "drop_topk", "drop_random_mean", "drop_random_std",
                "margin", "passed"} <= set(row)
    assert 0.0 <= res["base_score"] <= 1.0


def test_verdict_uses_the_aggregate_not_a_vote_per_k(corner_case):
    """At a small k a saturated softmax does not move for either mask, so that
    k carries no evidence. Counting it as a failure would sink a faithful map,
    so the verdict is the mean margin across k."""
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    res = fidelity_test(model, x, cam, class_idx=cls,
                        k_fractions=(0.05, 0.10, 0.20), n_random=5)
    assert res["passed"] is True
    assert res["n_k_passed"] < res["n_k"]       # a minority of k, yet it passes


def test_is_deterministic_for_a_fixed_seed(corner_case):
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    a = fidelity_test(model, x, cam, class_idx=cls, k_fractions=(0.1,), n_random=4, seed=7)
    b = fidelity_test(model, x, cam, class_idx=cls, k_fractions=(0.1,), n_random=4, seed=7)
    assert a["mean_margin"] == b["mean_margin"]


def test_figure_is_written(tmp_path, corner_case):
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    res = fidelity_test(model, x, cam, class_idx=cls, k_fractions=(0.05, 0.2), n_random=3)
    path = tmp_path / "fid.png"
    plot_fidelity(res, path=path, title="corner")
    assert path.exists() and path.stat().st_size > 0
