"""Stage 6.4 — split conformal prediction."""
import numpy as np
import pytest

from src.conformal import (calibrate, coverage, coverage_curve, nonconformity,
                           predict_set, set_sizes, summarize_sets)


def _synthetic(n, seed, noise=0.25):
    """Exchangeable binary data with a noisy but informative score."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, noise, size=n), 0.01, 0.99)
    return p, y


def test_nonconformity_accepts_matrix_and_vector():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = [0, 1]
    assert np.allclose(nonconformity(probs, y), [0.2, 0.3])
    assert np.allclose(nonconformity(probs[:, 1], y), [0.2, 0.3])   # p(tumor) vector


def test_calibrate_matches_the_finite_sample_quantile_by_hand():
    # scores: 0.1 0.2 0.3 0.4 0.5 (n=5)
    p_tumor = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.ones(5, int)
    # alpha=0.4 -> k = ceil(6*0.6) = 4 -> the 4th smallest score = 0.4
    assert calibrate(p_tumor, y, alpha=0.4) == pytest.approx(0.4)
    # alpha=0.2 -> k = ceil(6*0.8) = 5 = n -> the largest score = 0.5
    assert calibrate(p_tumor, y, alpha=0.2) == pytest.approx(0.5)
    # alpha=0.1 -> k = ceil(6*0.9) = 6 > n -> cannot be supported -> 1.0, every class in the set
    assert calibrate(p_tumor, y, alpha=0.1) == 1.0


def test_qhat_is_monotone_in_alpha():
    p, y = _synthetic(300, seed=1)
    qs = [calibrate(p, y, alpha=a) for a in (0.05, 0.1, 0.2, 0.3)]
    assert qs == sorted(qs, reverse=True)


def test_calibrate_rejects_bad_alpha_and_handles_empty():
    with pytest.raises(ValueError):
        calibrate([0.5], [1], alpha=0.0)
    assert calibrate([], [], alpha=0.1) == 1.0


def test_predict_set_shapes_and_edge_thresholds():
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    assert predict_set(probs, qhat=0.3) == [[0], [], [1]]
    assert predict_set(probs, qhat=1.0) == [[0, 1], [0, 1], [0, 1]]   # q=1 -> every class
    assert predict_set(probs[:, 1], qhat=0.3) == [[0], [], [1]]       # vector form


def test_coverage_by_hand():
    sets = [[0], [0, 1], [], [1]]
    assert coverage(sets, [0, 1, 1, 0]) == pytest.approx(0.5)
    assert np.isnan(coverage([], []))


def test_summarize_sets_reports_abstention():
    sets = [[0], [0, 1], [], [1]]
    s = summarize_sets(sets, [0, 1, 1, 1])
    assert s["n"] == 4 and s["coverage"] == 0.75
    assert s["frac_singleton"] == 0.5 and s["frac_abstain"] == 0.5 and s["frac_empty"] == 0.25
    assert set_sizes(sets).tolist() == [1, 2, 0, 1]


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_holds_on_exchangeable_data(alpha):
    """The one property that makes conformal prediction worth having: calibrate
    on one exchangeable sample, cover >= 1 - alpha on another, whatever the
    model's quality. Averaged over several seeds to keep the test stable."""
    covs = []
    for seed in range(8):
        pc, yc = _synthetic(200, seed=seed)
        pt, yt = _synthetic(400, seed=100 + seed)
        q = calibrate(pc, yc, alpha=alpha)
        covs.append(coverage(predict_set(pt, q), yt))
    assert np.mean(covs) >= 1 - alpha - 0.02


def test_coverage_curve_rows_are_well_formed():
    pc, yc = _synthetic(120, seed=3)
    pt, yt = _synthetic(120, seed=4)
    rows = coverage_curve(pc, yc, pt, yt, alphas=[0.1, 0.3])
    assert [r["alpha"] for r in rows] == [0.1, 0.3]
    assert all(0 <= r["coverage"] <= 1 and 0 <= r["mean_set_size"] <= 2 for r in rows)
    assert rows[0]["q_hat"] >= rows[1]["q_hat"]
