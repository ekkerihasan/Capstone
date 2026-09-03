"""A synthetic MSD-shaped dataset so stage 6.1 can be tested on any machine.

The real Task07_Pancreas tar is ~12 GB and lives only on the lab PC, so the
tests build a tiny phantom with the same on-disk layout, the same label
convention (0 background / 1 pancreas / 2 tumor) and deliberately *wrong*
spacing and orientation, which is what the preprocessing has to fix.
"""
import json

import nibabel as nib
import numpy as np
import pytest

SHAPE = (64, 64, 40)
RAW_SPACING = (0.8, 0.8, 2.5)  # not the target 1.5/1.5/2.0 -> Spacingd must resample


def _phantom(seed: int, with_tumor: bool):
    """A body ellipse of soft tissue on an air background, a pancreas blob
    inside it, and optionally a small tumor inside the pancreas."""
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


@pytest.fixture(scope="session")
def msd_phantom(tmp_path_factory):
    """Returns the root of a directory that looks like Task07_Pancreas."""
    root = tmp_path_factory.mktemp("Task07_Phantom")
    (root / "imagesTr").mkdir()
    (root / "labelsTr").mkdir()

    affine = np.diag(RAW_SPACING + (1.0,))
    affine[0, 0] *= -1  # LAS-ish, so Orientationd(RAS) has real work to do

    training = []
    for i in range(6):
        name = f"pancreas_{i:03d}.nii.gz"
        image, label = _phantom(seed=i, with_tumor=(i % 2 == 0))
        nib.save(nib.Nifti1Image(image, affine), root / "imagesTr" / name)
        nib.save(nib.Nifti1Image(label, affine), root / "labelsTr" / name)
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})

    # the macOS resource-fork junk that ships inside the official MSD tar
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
