"""Stage 6.6 - the abstention rule.

This is the module that decides whether a scan goes back to a radiologist, so
every branch is pinned by hand rather than by a round trip through a model.
"""
import pandas as pd
import pytest

from src.selective import (ABSTAIN, NO_TUMOR, TUMOR, decide, decide_frame,
                           label_of, summarize)


def test_confident_positive_and_negative():
    assert decide(0.9, conformal_set=[1])["decision"] == TUMOR
    assert decide(0.1, conformal_set=[0])["decision"] == NO_TUMOR


def test_threshold_is_inclusive_at_the_boundary():
    assert decide(0.5, conformal_set=[1])["decision"] == TUMOR
    assert decide(0.4999, conformal_set=[0])["decision"] == NO_TUMOR


def test_two_label_set_escalates_however_confident_the_probability_looks():
    """The whole point of the conformal set: a 0.97 probability with both labels
    inside the 90 % set is not a confident answer."""
    v = decide(0.97, conformal_set=[0, 1])
    assert v["decision"] == ABSTAIN
    assert "2 labels" in v["reasons"][0]


def test_empty_set_escalates():
    v = decide(0.6, conformal_set=[])
    assert v["decision"] == ABSTAIN
    assert "empty" in v["reasons"][0]


def test_mc_spread_escalates():
    v = decide(0.9, conformal_set=[1], mc_std=0.4)
    assert v["decision"] == ABSTAIN
    assert "disagree" in v["reasons"][0]
    assert decide(0.9, conformal_set=[1], mc_std=0.01)["decision"] == TUMOR


def test_failed_fidelity_escalates_but_absent_fidelity_does_not():
    """Stage 6.5 does not exist yet, so the signal is None everywhere. None must
    mean 'no evidence', not 'failed' - otherwise the app abstains on everything
    for the wrong reason."""
    assert decide(0.9, conformal_set=[1], fidelity_pass=False)["decision"] == ABSTAIN
    assert decide(0.9, conformal_set=[1], fidelity_pass=None)["decision"] == TUMOR
    assert decide(0.9, conformal_set=[1], fidelity_pass=True)["decision"] == TUMOR


def test_reasons_accumulate():
    v = decide(0.9, conformal_set=[0, 1], mc_std=0.4, fidelity_pass=False)
    assert len(v["reasons"]) == 3


def test_degrades_to_a_plain_threshold_with_no_signals():
    """Before 6.4 landed the pipeline was a bare threshold; with every optional
    signal absent the rule must still be exactly that."""
    assert decide(0.8)["decision"] == TUMOR
    assert decide(0.2)["decision"] == NO_TUMOR


def test_labels_phrase_abstention_as_an_action():
    assert label_of(ABSTAIN) == "FLAGGED FOR SPECIALIST REVIEW"
    assert label_of(TUMOR) == "Tumor present"
    assert label_of(NO_TUMOR) == "No tumor detected"


def test_decide_frame_adds_columns_and_parses_csv_sets():
    df = pd.DataFrame({
        "case": ["a", "b", "c"],
        "prob_tumor": [0.9, 0.2, 0.8],
        "conformal_set": ["[1]", "[0]", "[0, 1]"],   # as they come back off disk
        "mc_std": [0.01, 0.01, 0.01],
    })
    out = decide_frame(df)
    assert list(out["decision"]) == [TUMOR, NO_TUMOR, ABSTAIN]
    assert list(out["abstain"]) == [False, False, True]
    assert out.loc[2, "reasons"] != ""


def test_summarize_scores_only_the_answered_cases():
    """Accuracy over everything would hide the abstention; the selective
    headline is accuracy on what was answered, beside the refusal rate."""
    df = pd.DataFrame({
        "prob_tumor": [0.9, 0.1, 0.8, 0.9],
        "conformal_set": [[1], [0], [0, 1], [1]],
        "y_true": [1, 0, 1, 0],
    })
    s = summarize(decide_frame(df))
    assert s["n"] == 4 and s["n_answered"] == 3
    assert s["abstention_rate"] == pytest.approx(0.25)
    # summarize rounds to 4 dp; the last answered case is wrong, so 2 of 3
    assert s["accuracy_on_answered"] == pytest.approx(2 / 3, abs=1e-4)


def test_summarize_handles_an_empty_table():
    assert summarize(pd.DataFrame(columns=["abstain"]))["n"] == 0
