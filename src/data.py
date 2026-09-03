"""Stage 6.1 — dataset discovery and patient-wise splitting.

Every MSD Task07 case is one patient, so a case-level split *is* a patient-wise
split. Splits are written to JSON and reused so that stages 6.2-6.6 all score on
the same held-out patients.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from .config import MSD_DIR, SEED


def _clean(rel: str) -> str:
    return rel.lstrip("./").replace("\\", "/")


def case_id(path: str | os.PathLike) -> str:
    """'.../imagesTr/pancreas_004.nii.gz' -> 'pancreas_004'."""
    return Path(path).name.replace(".nii.gz", "").replace(".nii", "")


def load_msd_datalist(root: str | os.PathLike = MSD_DIR, section: str = "training"):
    """Read dataset.json and return [{'image': abs, 'label': abs}, ...].

    macOS resource-fork files ('._pancreas_004.nii.gz') ship inside the official
    MSD tar and are not readable volumes; they are filtered out here.
    """
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    cases = []
    for entry in meta[section]:
        if isinstance(entry, str):  # 'test' section is a bare list of images
            entry = {"image": entry}
        image = root / _clean(entry["image"])
        if image.name.startswith("._"):
            continue
        case = {"image": str(image)}
        if "label" in entry:
            case["label"] = str(root / _clean(entry["label"]))
        cases.append(case)
    return sorted(cases, key=lambda c: c["image"])


def patient_wise_split(cases, fractions=(0.7, 0.15, 0.15), names=("train", "val", "calib"), seed: int = SEED):
    """Shuffle patients once with a fixed seed and cut them into disjoint groups.

    Returns {name: [case, ...]}. No patient appears in two groups because the
    split is over cases and one case is one patient.
    """
    if len(fractions) != len(names):
        raise ValueError("fractions and names must be the same length")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")

    ordered = sorted(cases, key=lambda c: case_id(c["image"]))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    split, start = {}, 0
    for i, (name, frac) in enumerate(zip(names, fractions)):
        stop = n if i == len(names) - 1 else start + int(round(frac * n))
        split[name] = ordered[start:stop]
        start = stop
    return split


def save_split(split, path):
    """Write a split to JSON, storing case ids alongside paths so the file stays
    readable when the dataset moves to another machine."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "counts": {k: len(v) for k, v in split.items()},
        "case_ids": {k: [case_id(c["image"]) for c in v] for k, v in split.items()},
        "files": split,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_split(path, root: str | os.PathLike | None = None):
    """Read a split back. If `root` is given, rebase the stored paths onto it so
    a split made in Colab can be reused on the lab PC."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = payload["files"]
    if root is not None:
        root = Path(root)
        split = {
            k: [{kk: str(root / Path(vv).name) for kk, vv in c.items()} for c in v]
            for k, v in split.items()
        }
    return split


def assert_disjoint(split):
    """Guard against patient leakage between groups."""
    seen = {}
    for name, cases in split.items():
        for c in cases:
            cid = case_id(c["image"])
            if cid in seen:
                raise AssertionError(f"patient {cid} is in both '{seen[cid]}' and '{name}'")
            seen[cid] = name
    return True
