"""Generate the Review 2 test report from an actual pytest run.

"Testing so far" is an explicit Review 2 deliverable, so the report has to be
evidence rather than prose: this runs the suite, parses the JUnit XML it emits
and writes a table of what is covered and what passed. Regenerating it is the
only way to change it, which means the report cannot drift away from the suite.

    python -m src.run_test_report

Writes results/review2_demo/test_report.{md,csv}.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

from .config import REPO_ROOT, RESULTS_DIR

# Stage for the modules whose docstring does not name one.
STAGE_FALLBACK = {
    "test_metrics": "all",
    "test_pants_meta": "EDA",
}


def module_docstring(path: Path) -> str:
    """First line of the test module's docstring - its own description of what
    it covers. Reading it here means the report follows the tests, not a list
    maintained by hand."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (OSError, SyntaxError):
        return ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    # 'Stage 6.1 - dataset discovery...' -> 'dataset discovery...'
    return re.sub(r"^Stage\s+\d\.\d\s*[-–—:]\s*", "", first)


def module_stage(path: Path, doc: str) -> str:
    full = path.read_text(encoding="utf-8")[:400]
    m = re.search(r"Stage\s+(\d\.\d)", full)
    if m:
        return m.group(1)
    return STAGE_FALLBACK.get(path.stem, "-")


def run_pytest(xml_path: Path, extra_args) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}", *extra_args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def parse_junit(xml_path: Path):
    """-> (per-module dict, list of per-test rows, totals dict)."""
    root = ET.parse(xml_path).getroot()
    cases = root.iter("testcase")

    modules = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0,
                                   "skipped": 0, "time": 0.0})
    rows = []
    for c in cases:
        # classname is 'tests.test_transforms' or 'tests.test_x.TestClass'
        parts = c.get("classname", "").split(".")
        mod = next((p for p in parts if p.startswith("test_")), "unknown")
        status = "passed"
        if c.find("failure") is not None or c.find("error") is not None:
            status = "failed"
        elif c.find("skipped") is not None:
            status = "skipped"

        t = float(c.get("time", 0.0))
        m = modules[mod]
        m["total"] += 1
        m[status] += 1
        m["time"] += t
        rows.append({"module": mod, "test": c.get("name", ""),
                     "status": status, "time_s": round(t, 3)})

    totals = {k: sum(v[k] for v in modules.values())
              for k in ("total", "passed", "failed", "skipped")}
    totals["time"] = sum(v["time"] for v in modules.values())
    return modules, rows, totals


def write_report(modules, rows, totals, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = REPO_ROOT / "tests"

    lines = [
        "# Test Report",
        "",
        "**AI-Based Pancreatic Cancer Prediction Using Deep Learning "
        "(Early Detection from Medical Imaging Data)**",
        f"Generated {date.today().isoformat()} by `python -m src.run_test_report` "
        "- do not edit by hand, regenerate it.",
        "",
        f"**{totals['passed']} of {totals['total']} tests pass** "
        f"({totals['failed']} failed, {totals['skipped']} skipped) "
        f"in {totals['time']:.1f} s.",
        "",
        "The whole suite runs on CPU against the synthetic phantom in `src/phantom.py`, so it",
        "needs neither the 12 GB MSD download nor a GPU. That is the point: every stage stays",
        "verifiable on a laptop while the lab RTX 4060 is unavailable.",
        "",
        "| Module | Stage | Covers | Tests | Passed | Failed | Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]

    for mod in sorted(modules):
        m = modules[mod]
        path = tests_dir / f"{mod}.py"
        doc = module_docstring(path)
        stage = module_stage(path, doc) if path.exists() else "-"
        lines.append(
            f"| `{mod}.py` | {stage} | {doc or '-'} | {m['total']} | "
            f"{m['passed']} | {m['failed']} | {m['time']:.1f} |"
        )

    lines += [
        "",
        "## What the suite does not cover",
        "",
        "Stated so the table is not read as more than it is:",
        "",
        "- **No real-data result.** Every number here comes from the phantom. Dice and AUROC",
        "  measured on it are pipeline demonstrations, not results.",
        "- **No convergence claim.** `test_train_seg.py` checks that the loop, the metric and",
        "  the crash-resume path behave, not that the model reaches the Dice >= 0.75 target.",
        "- **No memory check.** Phantoms are 64x64x40; the 8 GB envelope at 96^3 patches is",
        "  exercised only on the lab PC.",
        "- **No calibration claim.** Synthetic logits look well calibrated because they were",
        "  generated; real ECE is measured on real predictions.",
        "",
        "## Per-test detail",
        "",
        "| Module | Test | Status | Time (s) |",
        "|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["module"], r["test"])):
        lines.append(f"| `{r['module']}` | `{r['test']}` | {r['status']} | {r['time_s']} |")

    md = out_dir / "test_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with (out_dir / "test_report.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["module", "test", "status", "time_s"])
        w.writeheader()
        w.writerows(rows)
    return md


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(RESULTS_DIR / "review2_demo"))
    p.add_argument("pytest_args", nargs="*", help="extra arguments passed to pytest")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        code = run_pytest(xml, args.pytest_args)
        if not xml.exists():
            print("pytest produced no XML; aborting", file=sys.stderr)
            return code or 1
        modules, rows, totals = parse_junit(xml)

    md = write_report(modules, rows, totals, Path(args.out))
    print(f"\n{totals['passed']}/{totals['total']} passed -> {md}")
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
