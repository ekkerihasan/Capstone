"""A synthetic MSD-shaped dataset, importable by scripts and notebooks.

The real Task07_Pancreas tar is ~12 GB and the GPU that trains on it is the lab
RTX 4060, so every stage has to be runnable without either. This module builds a
tiny phantom with the same on-disk layout as MSD, the same label convention
(0 background / 1 pancreas / 2 tumor) and deliberately *wrong* spacing and
orientation, which is what stage 6.1 preprocessing has to fix.

Two properties matter beyond "it runs":

- Half the cases carry no tumor, so the stage 6.3 present/absent classifier has
  negatives. Real MSD has a tumor in every case; the phantom does not.
- The tumor is near-isodense (35 HU inside a 55 HU pancreas), as in real PDAC,
  so the task is not trivially separable on intensity alone.

Anything measured on this data is a pipeline demonstration, never a result.
Figures derived from it must carry the stamp in `DEMO_STAMP`.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (64, 64, 40)
RAW_SPACING = (0.8, 0.8, 2.5)  # not the target 1.5/1.5/2.0 -> Spacingd must resample

DEMO_STAMP = "SYNTHETIC PHANTOM - PIPELINE DEMONSTRATION, NOT A RESULT"


def make_phantom(seed: int, with_tumor: bool):
    """A body ellipse of soft tissue on an air background, a pancreas blob
    inside it, and optionally a small tumor inside the pancreas.

    Returns (image, label) as float32 HU and uint8 labels, both of shape SHAPE.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")

    image = np.full(SHAPE, -1000.0, dtype=np.float32)          # air
    body = ((zz - 32) / 26) ** 2 + ((yy - 32) / 22) ** 2 < 1.0
    image[body] = 40.0 + rng.normal(0, 5, size=body.sum())     # soft tissue

    label = np.zeros(SHAPE, dtype=np.uint8)
    pancreas = ((zz - 28) / 9) ** 2 + ((yy - 30) / 5) ** 2 + ((xx - 20) / 7) ** 2 < 1.0
    label[pancreas] = 1
    image[pancreas] = 55.0 + rng.normal(0, 3, size=pancreas.sum())

    if with_tumor:
        tumor = ((zz - 28) / 3) ** 2 + ((yy - 30) / 3) ** 2 + ((xx - 20) / 3) ** 2 < 1.0
        tumor &= pancreas
        label[tumor] = 2
        image[tumor] = 35.0 + rng.normal(0, 3, size=tumor.sum())  # near-isodense, as in real PDAC

    return image, label


def has_tumor(index: int) -> bool:
    """Which cases carry a tumor. Alternating, so any contiguous slice of the
    dataset is roughly balanced for the stage 6.3 classifier."""
    return index % 2 == 0


def write_phantom_dataset(root, n_cases: int = 6, include_junk: bool = True) -> Path:
    """Write an MSD-shaped dataset to `root` and return it.

    `include_junk` adds the macOS resource-fork file ('._pancreas_000.nii.gz')
    that ships inside the official MSD tar, so the loader's skip logic is
    exercised here exactly as it is on the real data.
    """
    root = Path(root)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    affine = np.diag(RAW_SPACING + (1.0,))
    affine[0, 0] *= -1  # LAS-ish, so Orientationd(RAS) has real work to do

    training = []
    for i in range(n_cases):
        name = f"pancreas_{i:03d}.nii.gz"
        image, label = make_phantom(seed=i, with_tumor=has_tumor(i))
        nib.save(nib.Nifti1Image(image, affine), root / "imagesTr" / name)
        nib.save(nib.Nifti1Image(label, affine), root / "labelsTr" / name)
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})

    if include_junk:
        (root / "imagesTr" / "._pancreas_000.nii.gz").write_bytes(b"\x00\x05\x16\x07junk")
        training.append({"image": "./imagesTr/._pancreas_000.nii.gz",
                         "label": "./labelsTr/._pancreas_000.nii.gz"})

    (root / "dataset.json").write_text(json.dumps({
        "name": "Pancreas_Phantom",
        "labels": {"0": "background", "1": "pancreas", "2": "cancer"},
        "numTraining": len(training),
        "numTest": 0,
        "training": training,
        "test": [],
    }, indent=1), encoding="utf-8")
    return root


def stamp_figure(fig, text: str = DEMO_STAMP) -> None:
    """Stamp a matplotlib figure as phantom-derived.

    Every figure built from this data goes through here. An unlabelled phantom
    number in a review deck puts every other claim in doubt.
    """
    fig.text(0.5, 0.01, text, ha="center", va="bottom",
             fontsize=8, color="crimson", alpha=0.85)
