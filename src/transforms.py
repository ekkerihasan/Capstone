"""Stage 6.1 — preprocessing and augmentation transforms.

The pipeline is exactly the one specified in Chapter 6.1 of the report:

    | Report step                | Implementation                                  |
    |----------------------------|-------------------------------------------------|
    | Uniform voxel spacing      | Spacingd(1.5, 1.5, 2.0 mm)                      |
    | Consistent orientation     | Orientationd(RAS)                               |
    | HU clip to soft-tissue win | ScaleIntensityRanged(-100 ... 240 HU -> [0, 1]) |
    | Intensity normalisation    | same transform (min-max to [0, 1])              |
    | Augmentation (train only)  | random flip, 90 deg rotation, elastic deform    |
"""
from __future__ import annotations

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from .config import HU_MAX, HU_MIN, ORIENTATION, PATCH_SIZE, SPACING_MM

KEYS = ("image", "label")


def base_transforms(keys=KEYS, spacing=SPACING_MM, crop_foreground=True):
    """Deterministic part of stage 6.1 — applied to train and validation alike."""
    keys = list(keys)
    tf = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=ORIENTATION),
        Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
    if crop_foreground:
        # After windowing, air is exactly 0, so the body is the non-zero region.
        tf.append(CropForegroundd(keys=keys, source_key="image"))
    tf.append(EnsureTyped(keys=keys))
    return tf


def augment_transforms(keys=KEYS, elastic_prob=0.2):
    """Train-only augmentation. Elastic deformation is the expensive one, hence
    the low default probability; the Review-1 notebook forces it to 1.0 purely
    so the effect is visible in a single figure."""
    keys = list(keys)
    return [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        Rand3DElasticd(
            keys=keys,
            prob=elastic_prob,
            sigma_range=(5, 8),
            magnitude_range=(50, 120),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ]


def patch_transform(keys=KEYS, patch_size=PATCH_SIZE, num_samples=1, pos=2.0, neg=1.0):
    """Positive-biased 96^3 patch sampling — what stage 6.2 trains on inside the
    8 GB VRAM budget. Tumor voxels are rare, so patches are drawn 2:1 in favour
    of foreground.

    The pad in front is not optional: after `CropForegroundd` a slim patient can
    end up smaller than 96 voxels in one axis, and `RandCropByPosNegLabeld` then
    raises "proposed random crop ROI is larger than the image size" mid-epoch.
    """
    keys = list(keys)
    return Compose([
        SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant", constant_values=0),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key="image",
        ),
    ])


def build_transforms(
    train: bool,
    keys=KEYS,
    spacing=SPACING_MM,
    patches: bool = False,
    patch_size=PATCH_SIZE,
    elastic_prob: float = 0.2,
    num_samples: int = 1,
) -> Compose:
    """Assemble the stage 6.1 pipeline.

    train    : add augmentation (flip / rot90 / elastic)
    patches  : sample 96^3 patches instead of returning the whole volume

    Patch sampling comes *before* augmentation on purpose: `RandRotate90d` swaps
    the two in-plane axes, which changes the shape of a whole volume (a 26x23
    slice becomes 23x26) and breaks collation into a batch. Cubic patches are
    invariant to it.
    """
    tf = base_transforms(keys=keys, spacing=spacing)
    if patches:
        tf.append(patch_transform(keys=keys, patch_size=patch_size, num_samples=num_samples))
    if train:
        tf += augment_transforms(keys=keys, elastic_prob=elastic_prob)
    return Compose(tf)
