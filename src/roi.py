"""Stage 6.3, part one: pancreas ROI crop.

Takes a label mask and cuts a fixed-size window around the pancreas out of the
image. It does not care which model produced the mask: the ground-truth mask
(oracle path) and the stage 6.2 prediction (real path) go through the same code.

The pancreas ROI is every voxel labelled >= `label`, i.e. pancreas *and* tumor.
In MSD Task07 the tumor sits inside the organ but carries its own label (2), so
a bbox of label == 1 alone would clip the very lesion the classifier is meant to
see.

Spacing is never changed here. The window is cut at constant voxel size and
zero-padded where it runs off the volume, so the classifier sees the 1.5 x 1.5 x
2.0 mm grid that stage 6.1 produced, with no interpolation artefacts.

Signatures frozen in docs/REVIEW2_TEMP_PLAN.md section 9.
"""
from __future__ import annotations

import numpy as np

from .config import LABEL_PANCREAS, PATCH_SIZE, SPACING_MM


def _squeeze_mask(mask) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D (or 1xDxHxW), got shape {mask.shape}")
    return mask


def pancreas_bbox(mask, label=LABEL_PANCREAS, margin_mm=10.0, spacing_mm=SPACING_MM) -> tuple:
    """Bounding box of every voxel with value >= `label`, grown by `margin_mm`
    on each side and clipped to the volume. Returns a tuple of three slices.

    Raises ValueError on an empty mask: a case with no pancreas has no ROI, and
    silently returning the whole volume would feed the classifier background.
    """
    mask = _squeeze_mask(mask)
    foreground = mask >= label
    if not foreground.any():
        raise ValueError("mask contains no pancreas voxels; cannot build an ROI")

    slices = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        present = np.flatnonzero(foreground.any(axis=other))
        margin = int(np.ceil(margin_mm / float(spacing_mm[axis])))
        lo = max(int(present[0]) - margin, 0)
        hi = min(int(present[-1]) + 1 + margin, mask.shape[axis])
        slices.append(slice(lo, hi))
    return tuple(slices)


def _window(lo: int, hi: int, size: int) -> tuple[int, int]:
    """Start/stop of a `size`-long window centred on [lo, hi). May be negative or
    past the end; the caller pads."""
    start = (lo + hi) // 2 - size // 2
    return start, start + size


def _cut(volume: np.ndarray, starts, out_size, pad_value) -> np.ndarray:
    """volume[..., window] with constant padding outside the volume. Works on the
    last three axes, so a leading channel axis is carried through."""
    lead = volume.ndim - 3
    out = np.full(volume.shape[:lead] + tuple(out_size), pad_value, dtype=volume.dtype)
    src, dst = [], []
    for start, size, full in zip(starts, out_size, volume.shape[lead:]):
        lo, hi = max(start, 0), min(start + size, full)
        src.append(slice(lo, hi))
        dst.append(slice(lo - start, hi - start))
    out[(Ellipsis,) + tuple(dst)] = volume[(Ellipsis,) + tuple(src)]
    return out


def crop_roi(image, mask, out_size=PATCH_SIZE, margin_mm=10.0, spacing_mm=SPACING_MM) -> dict:
    """Fixed-size crop centred on the pancreas bbox.

    image : (D, H, W) or (C, D, H, W); mask : (D, H, W) or (1, D, H, W)
    Returns {"image", "mask", "bbox", "spacing_mm"}. `bbox` is the pancreas box
    (with margin) in source-volume coordinates, so a caller can tell whether the
    organ fit inside `out_size` or was truncated.
    """
    mask = _squeeze_mask(mask)
    image = np.asarray(image)
    if image.shape[-3:] != mask.shape:
        raise ValueError(f"image {image.shape} and mask {mask.shape} disagree on spatial shape")
    if len(out_size) != 3:
        raise ValueError(f"out_size must have 3 entries, got {out_size}")

    bbox = pancreas_bbox(mask, margin_mm=margin_mm, spacing_mm=spacing_mm)
    starts = [_window(s.start, s.stop, n)[0] for s, n in zip(bbox, out_size)]
    return {
        "image": _cut(image, starts, out_size, pad_value=0),
        "mask": _cut(mask, starts, out_size, pad_value=0),
        "bbox": bbox,
        "spacing_mm": tuple(spacing_mm),
    }


def bbox_fits(bbox, out_size) -> bool:
    """True when the whole pancreas box fits in the crop window."""
    return all((s.stop - s.start) <= n for s, n in zip(bbox, out_size))


# --------------------------------------------------------------------------
# Patch sampling: tumor vs non-tumor pancreatic tissue.
#
# Real MSD Task07 has a tumor in *every* case, so a case-level present/absent
# label has one class. The decision taken for the final review (docs/
# FINAL_2DAY_PLAN.md) is to learn tumor vs non-tumor *tissue* instead:
#   positive = a window containing the tumor centroid (jittered so the lesion
#              is not always dead centre - otherwise the network can learn
#              "blob in the middle" rather than "tumor");
#   negative = a window centred on pancreas tissue that contains no tumor
#              voxel at all, with a safety margin.
# PanTS stays external-validation-only, as CLAUDE.md requires.
# --------------------------------------------------------------------------

def _tumor_free(starts, out_size, tumor_lo, tumor_hi) -> bool:
    """True when the window [starts, starts+out_size) does not overlap the
    (margin-grown) tumor box [tumor_lo, tumor_hi) on at least one axis."""
    return any(s + n <= lo or s >= hi
               for s, n, lo, hi in zip(starts, out_size, tumor_lo, tumor_hi))


def sample_patch_starts(mask, out_size, n_neg: int, rng, tumor_label: int = 2,
                        pancreas_label: int = LABEL_PANCREAS, tumor_margin_mm: float = 5.0,
                        spacing_mm=SPACING_MM, max_tries: int = 200) -> list:
    """Window start corners for one positive (if the mask has a tumor) and up
    to `n_neg` tumor-free negatives. Returns [(starts, y), ...] with y in {0, 1}.

    Fewer than `n_neg` negatives come back when the pancreas has no room for a
    tumor-free window (a large tumor filling a small organ); never a negative
    that touches the tumor.
    """
    mask = _squeeze_mask(mask)
    size = np.asarray(out_size, dtype=int)
    out = []

    tumor = np.argwhere(mask == tumor_label)
    if len(tumor):
        lo_t, hi_t = tumor.min(0), tumor.max(0) + 1
        centroid = tumor.mean(0).round().astype(int)
        # jitter up to a quarter window, but keep the whole centroid inside
        jitter = rng.integers(-(size // 4), size // 4 + 1)
        out.append((tuple(int(v) for v in centroid + jitter - size // 2), 1))
        margin = np.ceil(tumor_margin_mm / np.asarray(spacing_mm, dtype=float)).astype(int)
        lo_t, hi_t = lo_t - margin, hi_t + margin
    else:
        lo_t = hi_t = None

    tissue = np.argwhere(mask == pancreas_label)
    if not len(tissue) or n_neg <= 0:
        return out
    found = []
    for _ in range(max_tries):
        if len(found) >= n_neg:
            break
        centre = tissue[rng.integers(len(tissue))]
        starts = tuple(int(v) for v in centre - size // 2)
        if lo_t is None or _tumor_free(starts, size, lo_t, hi_t):
            found.append((starts, 0))
    return out + found


def crop_at(volume, starts, out_size) -> np.ndarray:
    """Fixed-size window at `starts` (zero-padded off the volume edge)."""
    return _cut(np.asarray(volume), starts, out_size, pad_value=0)
