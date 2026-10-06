"""Stage 6.6: the abstention rule.

The project's promise is a prediction, a calibrated confidence and a verified
saliency map - *or* an explicit "flag for specialist review". This module is the
"or". It is deliberately a few lines of boolean logic with no model in it, so
the rule that decides whether a patient's scan goes back to a radiologist can be
read, argued about and tested on its own.

A case is escalated when any of the three signals says the model does not know:

    conformal set size != 1   the 90 % set holds both labels (or neither), so
                              the calibrated answer is "could be either"
    MC Dropout spread         the T=20 passes disagree with each other
    failed fidelity test      stage 6.5 could not confirm the saliency map
                              explains the prediction

Abstention is never an error in itself. The number to report is the accuracy on
the cases the model *did* answer, next to the fraction it refused.
"""
from __future__ import annotations

TUMOR = "tumor"
NO_TUMOR = "no_tumor"
ABSTAIN = "abstain"

THRESHOLD = 0.5
MAX_MC_STD = 0.15       # spread across MC Dropout passes, on the probability scale


def decide(prob_tumor: float, conformal_set=None, mc_std=None, fidelity_pass=None,
           threshold: float = THRESHOLD, max_mc_std: float = MAX_MC_STD) -> dict:
    """One case in, one decision out.

    Every signal is optional, because the stages that produce them land at
    different times: with none of them this degrades to a plain threshold, which
    is exactly what the pipeline did before stage 6.4 existed. Returns the
    decision, the reasons for it, and the signals it saw, so the UI can show the
    case *why* rather than just the verdict.
    """
    reasons = []

    if conformal_set is not None:
        n = len(conformal_set)
        if n > 1:
            reasons.append(f"conformal set holds {n} labels at the target coverage")
        elif n == 0:
            reasons.append("conformal set is empty: the case resembles no calibration case")

    if mc_std is not None and mc_std > max_mc_std:
        reasons.append(f"MC Dropout passes disagree (std {mc_std:.3f} > {max_mc_std:.3f})")

    if fidelity_pass is False:
        reasons.append("saliency map failed the F-Fidelity masking test")

    if reasons:
        decision = ABSTAIN
    else:
        decision = TUMOR if prob_tumor >= threshold else NO_TUMOR

    return {
        "decision": decision,
        "abstain": decision == ABSTAIN,
        "reasons": reasons,
        "prob_tumor": float(prob_tumor),
        "conformal_set": None if conformal_set is None else sorted(conformal_set),
        "mc_std": None if mc_std is None else float(mc_std),
        "fidelity_pass": fidelity_pass,
    }


def label_of(decision: str) -> str:
    """Wording for the UI and the report. The abstention is phrased as an action
    for a person, not as a failure of the model."""
    return {
        TUMOR: "Tumor present",
        NO_TUMOR: "No tumor detected",
        ABSTAIN: "FLAGGED FOR SPECIALIST REVIEW",
    }[decision]


def decide_frame(df, set_col: str = "conformal_set", prob_col: str = "prob_tumor",
                 std_col: str = "mc_std", **kw):
    """Apply `decide` across a predictions table and return it with decision,
    abstain and reason columns added. Missing signal columns are simply absent
    signals - the rule degrades instead of raising."""
    import pandas as pd

    rows = []
    for r in df.to_dict("records"):
        sets = r.get(set_col)
        if isinstance(sets, str):           # round-tripped through CSV
            sets = [int(c) for c in sets.replace("[", "").replace("]", "").split(",")
                    if c.strip() != ""]
        rows.append(decide(r[prob_col], conformal_set=sets,
                           mc_std=r.get(std_col), fidelity_pass=r.get("fidelity_pass"), **kw))
    out = pd.DataFrame(rows)[["decision", "abstain", "reasons"]]
    out["reasons"] = out["reasons"].map("; ".join)
    return pd.concat([df.reset_index(drop=True), out], axis=1)


def summarize(decided) -> dict:
    """Selective-prediction headline: what fraction was answered, and how good
    the answers were. Accuracy is computed on the answered cases only - that is
    the point of abstaining, and reporting it over everything would hide it."""
    n = len(decided)
    if n == 0:
        return {"n": 0}
    answered = decided[~decided["abstain"]]
    out = {
        "n": int(n),
        "n_answered": int(len(answered)),
        "abstention_rate": round(float(decided["abstain"].mean()), 4),
    }
    if "y_true" in decided.columns and len(answered):
        pred = (answered["decision"] == TUMOR).astype(int)
        out["accuracy_on_answered"] = round(float((pred == answered["y_true"]).mean()), 4)
    return out
