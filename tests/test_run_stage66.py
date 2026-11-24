"""Stage 6.6 case-level accepted accuracy and abstention summaries."""
import pandas as pd

from src.config import portable_path
from src.run_stage66 import summarize


def test_summary_paths_are_portable():
    assert portable_path("checkpoints/stage63/best.pt") == "checkpoints/stage63/best.pt"
    assert portable_path(r"D:\private\stage63\best.pt") == "best.pt"


def test_summarize_reports_accepted_accuracy_and_abstention_per_size_bucket():
    rows = pd.DataFrame([
        {"size_bucket": "sub2cm", "abstain": False, "correct": 1},
        {"size_bucket": "sub2cm", "abstain": True, "correct": None},
        {"size_bucket": "ge2cm", "abstain": False, "correct": 0},
        {"size_bucket": "no_tumor", "abstain": False, "correct": 1},
    ])

    result = summarize(rows)

    assert result["overall"] == {
        "n_cases": 4, "accepted": 3, "abstention_rate": 0.25, "accepted_accuracy": 0.6667,
    }
    assert result["sub2cm"] == {
        "n_cases": 2, "accepted": 1, "abstention_rate": 0.5, "accepted_accuracy": 1.0,
    }
    assert result["ge2cm"]["accepted_accuracy"] == 0.0
    assert result["no_tumor"]["abstention_rate"] == 0.0
