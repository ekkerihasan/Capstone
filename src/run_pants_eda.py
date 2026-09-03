"""PanTS metadata EDA — the "PanTS at a glance" slide, with real numbers.

    python -m src.run_pants_eda --out results/review1

Writes to --out:
    pants_lesions.csv          one row per measured pancreas lesion
    pants_eda_summary.json     cohort counts, measurement coverage, sub-2 cm tally
    fig_pants_overview.png     phase / site / tumor prevalence
    fig_pants_tumor_size.png   lesion long-axis distribution against the 2 cm line

Runs on CPU in seconds. Only needs data/PanTS/metadata.xlsx (~1.3 MB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, SMALL_TUMOR_CM
from .pants_meta import (METADATA_XLSX, benign_lesions, coverage, lesion_table,
                         load_metadata, sub2cm_summary, tumor_lesions)

PURPLE = "#5b4fbf"
RED = "#d1495b"


def plot_overview(df, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    phase = df["ct phase"].value_counts()
    ax[0].bar(phase.index.astype(str), phase.values, color=PURPLE)
    ax[0].set_title(f"Contrast phase (n={len(df)})")
    ax[0].tick_params(axis="x", rotation=30)

    tumor = df["tumor?"].value_counts().sort_index()
    labels = ["no tumor", "tumor"][: len(tumor)]
    ax[1].bar(labels, tumor.values, color=[PURPLE, RED][: len(tumor)])
    pct = 100.0 * tumor.get(1, 0) / len(df)
    ax[1].set_title(f"Tumor prevalence — {tumor.get(1, 0)} cases ({pct:.1f}%)")
    for i, v in enumerate(tumor.values):
        ax[1].text(i, v, str(v), ha="center", va="bottom")

    site = df["site"].value_counts().head(15)
    ax[2].barh(site.index.astype(str)[::-1], site.values[::-1], color=PURPLE)
    ax[2].set_title(f"Scans per site (top 15 of {df['site'].nunique()})")
    ax[2].tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tumor_size(lesions, path, threshold=SMALL_TUMOR_CM):
    """Tumor-flagged lesions only. Benign lesions are shown as a separate series
    in the first panel rather than pooled into the tumor distribution."""
    import matplotlib.pyplot as plt

    measured = tumor_lesions(lesions)
    benign = benign_lesions(lesions)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    small = measured[measured["long_axis_cm"] < threshold]
    bins = np.linspace(0, max(8.0, float(measured["long_axis_cm"].max())), 33)
    ax[0].hist(benign["long_axis_cm"], bins=bins, color="#c9c6e0",
               label=f"benign lesions (n={len(benign)})")
    ax[0].hist(measured["long_axis_cm"], bins=bins, color=PURPLE, alpha=0.9,
               label=f"tumor lesions (n={len(measured)})")
    ax[0].axvline(threshold, color=RED, ls="--", lw=1.5)
    ax[0].set_title(f"Lesion long axis — tumors sub-{threshold:g} cm: "
                    f"{len(small)}/{len(measured)} ({100 * len(small) / len(measured):.1f}%)")
    ax[0].set_xlabel("cm")
    ax[0].set_ylabel("lesions")
    ax[0].legend(fontsize=8)

    if "enhancement" in measured:
        order = ["hypoattenuating", "isoattenuating", "hyperattenuating"]
        present = [o for o in order if o in set(measured["enhancement"].dropna())]
        under = [int(((measured["enhancement"] == e) & (measured["long_axis_cm"] < threshold)).sum())
                 for e in present]
        over = [int(((measured["enhancement"] == e) & (measured["long_axis_cm"] >= threshold)).sum())
                for e in present]
        x = np.arange(len(present))
        ax[1].bar(x, under, color=RED, label=f"< {threshold:g} cm")
        ax[1].bar(x, over, bottom=under, color=PURPLE, label=f">= {threshold:g} cm")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([p.replace("attenuating", "-att.") for p in present], rotation=15)
        ax[1].set_title("Enhancement vs. pancreas")
        ax[1].legend(fontsize=8)

    if "location" in measured:
        loc = measured["location"].value_counts().head(8)
        ax[2].barh(loc.index.astype(str)[::-1], loc.values[::-1], color=PURPLE)
        ax[2].set_title("Lesion location")
        ax[2].tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="PanTS metadata EDA")
    p.add_argument("--metadata", default=str(METADATA_XLSX))
    p.add_argument("--out", default=str(RESULTS_DIR / "review1"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata)
    lesions = lesion_table(df)
    lesions.to_csv(out / "pants_lesions.csv", index=False)

    cov = coverage(df, lesions)
    size = sub2cm_summary(lesions)
    summary = {
        "source": str(args.metadata),
        "cohort": cov,
        "tumor_size": size,
        "phases": df["ct phase"].value_counts(dropna=False).to_dict(),
        "n_sites": int(df["site"].nunique()),
        "age_median": float(df["age"].median(skipna=True)),
        "sub2cm_all_lesions_including_benign": sub2cm_summary(lesions, tumor_only=False),
        "note": ("Lesion sizes are parsed from the free-text structured reports; "
                 "metadata.xlsx has no tumor-size column. Two denominators matter: "
                 "(a) only tumor-flagged cases with a lesion block can be measured "
                 "(see measurement_coverage_pct), and (b) most measured lesions "
                 "belong to cases flagged tumor?=0 and are benign, so tumor_size "
                 "counts tumor-flagged cases only."),
    }
    summary["phases"] = {str(k): int(v) for k, v in summary["phases"].items()}
    (out / "pants_eda_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if not args.no_figures:
        plot_overview(df, out / "fig_pants_overview.png")
        plot_tumor_size(lesions, out / "fig_pants_tumor_size.png")

    print(f"[PanTS] {cov['total_cases']} cases | {cov['tumor_cases']} with tumor "
          f"({100 * cov['tumor_cases'] / cov['total_cases']:.1f}%)")
    print(f"[PanTS] tumor lesions measured: {cov['measured_tumor_lesions']} across "
          f"{cov['tumor_cases_with_measured_lesion']} cases "
          f"({cov['measurement_coverage_pct']}% of tumor cases)")
    print(f"[PanTS] benign lesions (tumor?=0), kept separate: "
          f"{cov['measured_benign_lesions']} across "
          f"{cov['benign_cases_with_measured_lesion']} cases")
    print(f"[PanTS] sub-{size['threshold_cm']:g} cm: {size['sub_threshold']}/{size['measured']} "
          f"({size['sub_threshold_pct']}%) | median long axis {size['median_long_axis_cm']} cm")
    if "sub_threshold_and_isoattenuating" in size:
        print(f"[PanTS] hardest stratum (sub-2 cm AND isoattenuating): "
              f"{size['sub_threshold_and_isoattenuating']} lesions")
    print(f"[PanTS] -> {out}")
    return summary


if __name__ == "__main__":
    main()
