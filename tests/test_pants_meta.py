"""PanTS structured-report parsing.

The fixtures below are real report text from metadata.xlsx, trimmed. The
`_x000D_` escapes are how Excel stores the newlines and must be handled.
"""
import pandas as pd
import pytest

from src.pants_meta import (benign_lesions, clean_report, coverage, lesion_table,
                            parse_report, sub2cm_summary, tumor_lesions)

ONE_LESION = (
    "FINDINGS: _x000D_ Pancreas: _x000D_ Normal size (volume: 77.0 cc)._x000D_ "
    "_x000D_ Pancreas lesions:_x000D_ Pancreas lesion 1: _x000D_ "
    "Location: pancreas head._x000D_ "
    "Size: 2.4 x 1.4 cm (image 14). Volume: 5.3 cc._x000D_ "
    "Enhancement relative to pancreas: Isoattenuating (HU value is 90.6+/-21.1)._x000D_ "
    "_x000D_ Kidney: _x000D_ Normal size (right kidney volume: 175.5 cc)._x000D_ "
    "IMPRESSION:_x000D_ A isoattenuating pancreas (head) mass (2.4 x 1.4 cm)."
)

TWO_LESIONS = (
    "Pancreas lesions:_x000D_ Pancreas lesion 1: _x000D_ Location: pancreas tail._x000D_ "
    "Size: 1.6 x 1.1 cm. Volume: 1.1 cc._x000D_ "
    "Enhancement relative to pancreas: Hypoattenuating._x000D_ "
    "Pancreas lesion 2: _x000D_ Location: pancreas head._x000D_ "
    "Size: 1.6 x 1.0 cm. Volume: 0.7 cc._x000D_ "
    "Enhancement relative to pancreas: Hyperattenuating._x000D_ Kidney: _x000D_ Normal size."
)

NO_LESION = (
    "FINDINGS: _x000D_ Spleen: _x000D_ Normal size (volume: 182.9 cc)._x000D_ "
    "Pancreas: _x000D_ Normal size (volume: 70.1 cc)._x000D_ IMPRESSION:_x000D_ Unremarkable."
)


def test_clean_report_expands_excel_newlines():
    assert "_x000D_" not in clean_report(ONE_LESION)
    assert clean_report(None) == ""
    assert clean_report(float("nan")) == ""


def test_parses_a_single_lesion_completely():
    (lesion,) = parse_report(ONE_LESION)
    assert lesion["lesion"] == 1
    assert lesion["dims_cm"] == [2.4, 1.4]
    assert lesion["long_axis_cm"] == 2.4
    assert lesion["short_axis_cm"] == 1.4
    assert lesion["volume_cc"] == 5.3
    assert lesion["location"] == "pancreas head"
    assert lesion["enhancement"] == "isoattenuating"


def test_organ_volume_is_not_mistaken_for_lesion_volume():
    """The report states a pancreas volume of 77.0 cc before the lesion block;
    the lesion's own volume is 5.3 cc."""
    (lesion,) = parse_report(ONE_LESION)
    assert lesion["volume_cc"] == 5.3


def test_kidney_volume_after_the_lesion_does_not_leak_in():
    """The lesion body must stop at the next organ heading, or the kidney's
    175.5 cc would be picked up by a later lesion field."""
    (lesion,) = parse_report(ONE_LESION)
    assert lesion["volume_cc"] != 175.5


def test_parses_multiple_lesions_separately():
    lesions = parse_report(TWO_LESIONS)
    assert [l["lesion"] for l in lesions] == [1, 2]
    assert [l["location"] for l in lesions] == ["pancreas tail", "pancreas head"]
    assert [l["enhancement"] for l in lesions] == ["hypoattenuating", "hyperattenuating"]
    assert [l["volume_cc"] for l in lesions] == [1.1, 0.7]


def test_report_without_a_lesion_block_yields_nothing():
    assert parse_report(NO_LESION) == []
    assert parse_report(None) == []
    assert parse_report("") == []


def test_millimetre_sizes_are_converted_to_centimetres():
    mm = "Pancreas lesion 1: Location: pancreas head. Size: 18.0 x 12.0 mm. Volume: 1.2 cc."
    (lesion,) = parse_report(mm)
    assert lesion["long_axis_cm"] == pytest.approx(1.8)
    assert lesion["short_axis_cm"] == pytest.approx(1.2)


def test_three_dimensional_size_uses_the_longest_axis():
    three = "Pancreas lesion 1: Size: 2.0 x 1.4 x 3.1 cm. Volume: 4.0 cc."
    (lesion,) = parse_report(three)
    assert lesion["dims_cm"] == [2.0, 1.4, 3.1]
    assert lesion["long_axis_cm"] == 3.1


# ------------------------------------------------------- tumor vs benign cohorts


@pytest.fixture
def cohort():
    return pd.DataFrame([
        {"PanTS ID": "A", "tumor?": 1, "structured report": ONE_LESION,
         "ct phase": "Venous", "site": "S1", "sex": "M", "age": 60},
        {"PanTS ID": "B", "tumor?": 1, "structured report": NO_LESION,
         "ct phase": "Venous", "site": "S1", "sex": "F", "age": 70},
        {"PanTS ID": "C", "tumor?": 0, "structured report": TWO_LESIONS,
         "ct phase": "Arterial", "site": "S2", "sex": "M", "age": 55},
    ])


def test_lesion_table_keeps_the_tumor_flag(cohort):
    table = lesion_table(cohort)
    assert len(table) == 3                      # 1 from A, 0 from B, 2 from C
    assert set(table["PanTS ID"]) == {"A", "C"}
    assert dict(zip(table["PanTS ID"], table["tumor"]))["C"] == 0


def test_tumor_and_benign_lesions_are_separated(cohort):
    table = lesion_table(cohort)
    assert len(tumor_lesions(table)) == 1        # only A
    assert len(benign_lesions(table)) == 2       # both from C


def test_coverage_denominator_is_tumor_cases_not_lesion_cases(cohort):
    """Regression: pooling benign lesions in reported 94% coverage when the
    real figure was 29% — the two populations must not be mixed."""
    cov = coverage(cohort, lesion_table(cohort))
    assert cov["tumor_cases"] == 2
    assert cov["tumor_cases_with_measured_lesion"] == 1
    assert cov["measurement_coverage_pct"] == 50.0
    assert cov["measured_tumor_lesions"] == 1
    assert cov["measured_benign_lesions"] == 2   # counted, but kept apart


def test_sub2cm_summary_defaults_to_tumor_cases_only(cohort):
    table = lesion_table(cohort)
    tumor_only = sub2cm_summary(table)
    assert tumor_only["measured"] == 1           # the 2.4 cm lesion in case A
    assert tumor_only["sub_threshold"] == 0
    assert tumor_only["population"] == "tumor-flagged cases"

    pooled = sub2cm_summary(table, tumor_only=False)
    assert pooled["measured"] == 3               # benign lesions pulled in
    assert pooled["sub_threshold"] == 2
    assert pooled["population"] == "all measured lesions"


def test_sub2cm_summary_reports_the_hardest_stratum(cohort):
    """Small AND isoattenuating is the stratum the project exists to address."""
    table = lesion_table(cohort)
    out = sub2cm_summary(table, threshold=3.0)
    assert out["sub_threshold_and_isoattenuating"] == 1
