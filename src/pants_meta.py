"""PanTS metadata — parsing the structured radiology reports.

`metadata.xlsx` has no tumor-size column. Lesion measurements live in free text
inside the `structured report` field, in blocks that look like:

    Pancreas lesions:
    Pancreas lesion 1:
    Location: pancreas head.
    Size: 2.4 x 1.4 cm (image 14). Volume: 5.3 cc.
    Enhancement relative to pancreas: Isoattenuating (HU value is 90.6+/-21.1).

This module pulls those out so the sub-2 cm question — the premise of the whole
project — can be answered with a real number instead of a guessed column.

Two caveats that must survive into any review slide:

1. Not every tumor-flagged case has a lesion block. `coverage()` reports how
   many could be measured so the denominator is never quietly dropped.
2. A "pancreas lesion" in these reports is NOT the same thing as the `tumor?`
   flag — most measured lesions sit in cases flagged `tumor? == 0` (benign
   cysts and the like). Pooling them inflates the cohort and skews the size
   distribution, so every function here keeps the two populations apart and
   defaults to the tumor-flagged one.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PANTS_DIR, SMALL_TUMOR_CM

METADATA_XLSX = PANTS_DIR / "metadata.xlsx"

# Excel stores the report's newlines as a literal "_x000D_" escape.
CARRIAGE = "_x000D_"

LESION_SPLIT = re.compile(r"Pancreas lesion\s+(\d+)\s*:", re.IGNORECASE)
SIZE = re.compile(
    r"Size:\s*([\d.]+)\s*x\s*([\d.]+)(?:\s*x\s*([\d.]+))?\s*(cm|mm)", re.IGNORECASE
)
VOLUME = re.compile(r"Volume:\s*([\d.]+)\s*cc", re.IGNORECASE)
LOCATION = re.compile(r"Location:\s*([^.\n]+)", re.IGNORECASE)
ENHANCEMENT = re.compile(
    r"Enhancement relative to pancreas:\s*([A-Za-z\-]+)", re.IGNORECASE
)


def clean_report(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).replace(CARRIAGE, "\n")


def parse_report(text) -> list[dict]:
    """Return one dict per pancreas lesion found in a structured report.

    Cases with no lesion block return an empty list, which is the honest answer
    for a report that never measured anything.
    """
    report = clean_report(text)
    if not report:
        return []

    parts = LESION_SPLIT.split(report)
    if len(parts) < 3:                       # no "Pancreas lesion N:" heading
        return []

    lesions = []
    # parts == [preamble, '1', body1, '2', body2, ...]
    for index, body in zip(parts[1::2], parts[2::2]):
        # A lesion body ends where the next organ section starts.
        body = re.split(r"\n\s*(?:Kidney|Liver|Spleen|IMPRESSION)\s*:", body)[0]
        lesion = {"lesion": int(index)}

        m = SIZE.search(body)
        if m:
            dims = [float(d) for d in m.group(1, 2, 3) if d is not None]
            if m.group(4).lower() == "mm":
                dims = [d / 10.0 for d in dims]
            lesion["dims_cm"] = dims
            lesion["long_axis_cm"] = max(dims)
            lesion["short_axis_cm"] = min(dims)

        m = VOLUME.search(body)
        if m:
            lesion["volume_cc"] = float(m.group(1))

        m = LOCATION.search(body)
        if m:
            lesion["location"] = m.group(1).strip().lower()

        m = ENHANCEMENT.search(body)
        if m:
            lesion["enhancement"] = m.group(1).strip().lower()

        lesions.append(lesion)
    return lesions


def load_metadata(path=METADATA_XLSX) -> pd.DataFrame:
    df = pd.read_excel(Path(path))
    df.columns = [c.strip() for c in df.columns]
    return df


def lesion_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per measured lesion, joined back to its case."""
    rows = []
    for _, case in df.iterrows():
        for lesion in parse_report(case.get("structured report")):
            rows.append({
                "PanTS ID": case.get("PanTS ID"),
                "tumor": case.get("tumor?"),
                "ct phase": case.get("ct phase"),
                "site": case.get("site"),
                "sex": case.get("sex"),
                "age": case.get("age"),
                **lesion,
            })
    out = pd.DataFrame(rows)
    if len(out) and "long_axis_cm" in out:
        out["sub2cm"] = out["long_axis_cm"] < SMALL_TUMOR_CM
    return out


def tumor_lesions(lesions: pd.DataFrame) -> pd.DataFrame:
    """Measured lesions in cases the dataset flags as having a tumor."""
    if not len(lesions):
        return lesions
    return lesions[(lesions["tumor"] == 1) & lesions["long_axis_cm"].notna()]


def benign_lesions(lesions: pd.DataFrame) -> pd.DataFrame:
    """Measured lesions in cases flagged `tumor? == 0` — a real confounder for
    the stage 6.3 classifier, not noise to be discarded."""
    if not len(lesions):
        return lesions
    return lesions[(lesions["tumor"] == 0) & lesions["long_axis_cm"].notna()]


def coverage(df: pd.DataFrame, lesions: pd.DataFrame) -> dict:
    """How much of the tumor cohort the parser could actually measure. Quote
    `measurement_coverage_pct` alongside any sub-2 cm percentage."""
    tumor_cases = int((df["tumor?"] == 1).sum())
    tumor = tumor_lesions(lesions)
    benign = benign_lesions(lesions)
    return {
        "total_cases": int(len(df)),
        "tumor_cases": tumor_cases,
        "tumor_cases_with_measured_lesion": int(tumor["PanTS ID"].nunique()) if len(tumor) else 0,
        "measured_tumor_lesions": int(len(tumor)),
        "measurement_coverage_pct": round(
            100.0 * (tumor["PanTS ID"].nunique() if len(tumor) else 0) / tumor_cases, 1
        ) if tumor_cases else 0.0,
        # kept separate on purpose — these are not tumors
        "benign_cases_with_measured_lesion": int(benign["PanTS ID"].nunique()) if len(benign) else 0,
        "measured_benign_lesions": int(len(benign)),
    }


def sub2cm_summary(lesions: pd.DataFrame, threshold: float = SMALL_TUMOR_CM,
                   tumor_only: bool = True) -> dict:
    """The headline number: how many measured lesions are under the threshold.

    Defaults to tumor-flagged cases only. Passing tumor_only=False pools benign
    lesions in and answers a different question — say which one you mean.
    """
    measured = tumor_lesions(lesions) if tumor_only else lesions.dropna(subset=["long_axis_cm"])
    if not len(measured):
        return {"measured": 0}
    small = measured[measured["long_axis_cm"] < threshold]
    out = {
        "population": "tumor-flagged cases" if tumor_only else "all measured lesions",
        "threshold_cm": threshold,
        "measured": int(len(measured)),
        "sub_threshold": int(len(small)),
        "sub_threshold_pct": round(100.0 * len(small) / len(measured), 1),
        "median_long_axis_cm": round(float(measured["long_axis_cm"].median()), 2),
        "min_long_axis_cm": round(float(measured["long_axis_cm"].min()), 2),
        "max_long_axis_cm": round(float(measured["long_axis_cm"].max()), 2),
    }
    if "enhancement" in measured:
        iso = measured[measured["enhancement"] == "isoattenuating"]
        out["isoattenuating"] = int(len(iso))
        # The project's hardest stratum: small AND near-isodense with pancreas.
        out["sub_threshold_and_isoattenuating"] = int(
            len(measured[(measured["long_axis_cm"] < threshold)
                         & (measured["enhancement"] == "isoattenuating")])
        )
    return out
