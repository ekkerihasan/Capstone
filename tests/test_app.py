"""Stage 6.6 - the demo UI's logic, without launching a server.

Gradio itself is not exercised here; what is, is everything the callback does
before it hands a figure and a verdict back to the browser. The app is the one
place a phantom number could reach a viewer unlabelled, so the provenance text
is tested as carefully as the arithmetic.
"""
import json

import numpy as np
import pytest
import torch

from src.app import (DEMO_STAMP, _tumor_probability, analyse, find_checkpoint,
                     list_cases, load_model, load_volume, q_hat_for,
                     report_markdown, training_patch)
from src.phantom import write_phantom_dataset
from src.selective import ABSTAIN


@pytest.fixture(scope="module")
def phantom_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("app_phantom")
    return write_phantom_dataset(root, n_cases=4, include_junk=True,
                                 pancreas_axes=(14.0, 9.0, 11.0), tumor_radii=(9.0,))


@pytest.fixture(scope="module")
def model():
    m, trained = load_model(None)       # random weights: the app must still run
    return m, trained


def test_list_cases_skips_the_macos_junk(phantom_root):
    cases = list_cases(phantom_root)
    assert len(cases) == 4
    assert not any(c.startswith("._") for c in cases)


def test_load_volume_applies_stage61_and_finds_the_label(phantom_root):
    x, gt = load_volume(phantom_root / "imagesTr" / "pancreas_000.nii.gz")
    assert x.ndim == 5 and x.shape[0] == 1          # [N, C, D, H, W]
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0   # HU window -> [0, 1]
    assert gt is not None and set(np.unique(gt)) <= {0, 1, 2}


def test_tumor_probability_is_bounded_and_responds_to_the_tumor_channel():
    quiet = torch.zeros(1, 3, 4, 4, 4)
    loud = quiet.clone()
    loud[0, 2] = 10.0                                # tumor channel dominates
    assert 0.0 <= _tumor_probability(quiet) <= 1.0
    assert _tumor_probability(loud) > _tumor_probability(quiet)


def test_analyse_returns_a_verdict_on_untrained_weights(phantom_root, model):
    m, trained = model
    fig, verdict, gt, is_trained, alpha = analyse(
        phantom_root / "imagesTr" / "pancreas_000.nii.gz", m, trained,
        q_hat=0.5, alpha=0.1, mc_passes=3, patch=(32, 32, 32))
    assert is_trained is False
    assert 0.0 <= verdict["prob_tumor"] <= 1.0
    assert verdict["mc_std"] >= 0.0
    assert verdict["decision"] in {"tumor", "no_tumor", ABSTAIN}
    assert fig is not None


def test_predict_restores_the_models_dropout_rates(phantom_root, model):
    """MC Dropout raises p for its passes; leaving it raised would quietly
    change the model for every later call."""
    m, trained = model
    before = {n: mod.p for n, mod in m.named_modules() if hasattr(mod, "p")}
    analyse(phantom_root / "imagesTr" / "pancreas_000.nii.gz", m, trained,
            q_hat=0.5, alpha=0.1, mc_passes=3, patch=(32, 32, 32))
    after = {n: mod.p for n, mod in m.named_modules() if hasattr(mod, "p")}
    assert before == after
    assert not m.training


def test_conformal_set_widens_as_q_hat_grows(phantom_root, model):
    m, trained = model
    path = phantom_root / "imagesTr" / "pancreas_000.nii.gz"
    small = analyse(path, m, trained, q_hat=0.05, alpha=0.5, mc_passes=1,
                    patch=(32, 32, 32))[1]
    large = analyse(path, m, trained, q_hat=1.0, alpha=0.02, mc_passes=1,
                    patch=(32, 32, 32))[1]
    assert len(large["conformal_set"]) >= len(small["conformal_set"])
    assert large["conformal_set"] == [0, 1]


def test_missing_q_hat_drops_the_signal_rather_than_raising(phantom_root, model):
    m, trained = model
    verdict = analyse(phantom_root / "imagesTr" / "pancreas_000.nii.gz", m, trained,
                      q_hat=None, alpha=None, mc_passes=1, patch=(32, 32, 32))[1]
    assert verdict["conformal_set"] is None


def test_q_hat_for_picks_the_nearest_calibrated_alpha():
    curve = [(0.02, 1.0), (0.10, 0.64), (0.30, 0.50)]
    assert q_hat_for(curve, 0.11) == 0.64
    assert q_hat_for(curve, 0.29) == 0.50
    assert q_hat_for([], 0.1, fallback=0.9) == 0.9


def test_report_always_declares_its_provenance():
    """Nothing in this app may reach a viewer without saying what it is."""
    verdict = {"decision": ABSTAIN, "abstain": True, "prob_tumor": 0.5,
               "mc_std": 0.3, "conformal_set": [0, 1],
               "reasons": ["conformal set holds 2 labels at the target coverage"],
               "fidelity_pass": None}
    md = report_markdown(verdict, gt=None, trained=False, alpha=0.1, mc_passes=20)
    assert DEMO_STAMP in md
    assert "UNTRAINED" in md
    assert "no stage 6.3 classifier exists yet" in md
    assert "stage 6.5 not built" in md
    assert "FLAGGED FOR SPECIALIST REVIEW" in md


def test_report_says_when_a_checkpoint_was_loaded():
    verdict = {"decision": "tumor", "abstain": False, "prob_tumor": 0.9,
               "mc_std": None, "conformal_set": None, "reasons": [],
               "fidelity_pass": None}
    md = report_markdown(verdict, gt=None, trained=True, alpha=0.1, mc_passes=20)
    assert "checkpoint loaded" in md and "UNTRAINED" not in md


def test_training_patch_reads_the_checkpoints_own_window(tmp_path):
    """A model trained on 32^3 patches evaluated at 96^3 is a different model."""
    summary = tmp_path / "stage62_summary.json"
    summary.write_text(json.dumps({"patch": [48, 48, 48]}), encoding="utf-8")
    assert training_patch(summary) == (48, 48, 48)
    assert training_patch(tmp_path / "absent.json") == (32, 32, 32)


def test_find_checkpoint_returns_none_when_there_is_nothing(tmp_path, monkeypatch):
    import src.app as app

    monkeypatch.setattr(app, "DEFAULT_CKPT", tmp_path / "nope.pt")
    monkeypatch.setattr(app, "CHECKPOINT_DIR", tmp_path)
    assert find_checkpoint("") is None
