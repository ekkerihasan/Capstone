"""Stage 6.3 ROI crop. Pure numpy: no MONAI, no data on disk."""
import numpy as np
import pytest

from src.roi import bbox_fits, crop_roi, pancreas_bbox

SP = (1.0, 1.0, 1.0)


def make_mask(shape=(40, 40, 40), box=((10, 20), (12, 18), (15, 25)), tumor_at=None):
    mask = np.zeros(shape, dtype=np.uint8)
    (a, b), (c, d), (e, f) = box
    mask[a:b, c:d, e:f] = 1
    if tumor_at is not None:
        z, y, x = tumor_at
        mask[z, y, x] = 2
    return mask


def test_bbox_is_tight_without_margin():
    bbox = pancreas_bbox(make_mask(), margin_mm=0.0, spacing_mm=SP)
    assert bbox == (slice(10, 20), slice(12, 18), slice(15, 25))


def test_margin_is_in_mm_not_voxels():
    # 10 mm at 2 mm/voxel is 5 voxels on that axis, 10 on the 1 mm axes
    bbox = pancreas_bbox(make_mask(), margin_mm=10.0, spacing_mm=(1.0, 1.0, 2.0))
    assert bbox[0] == slice(0, 30)       # 10-10 .. 20+10
    assert bbox[2] == slice(10, 30)      # 15-5  .. 25+5


def test_bbox_is_clipped_to_the_volume():
    bbox = pancreas_bbox(make_mask(), margin_mm=100.0, spacing_mm=SP)
    assert bbox == (slice(0, 40), slice(0, 40), slice(0, 40))


def test_tumor_voxels_count_as_pancreas():
    # a tumor (label 2) sticking out of the label-1 region must stay inside the box
    mask = make_mask(tumor_at=(30, 14, 16))
    bbox = pancreas_bbox(mask, margin_mm=0.0, spacing_mm=SP)
    assert bbox[0].stop == 31


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="no pancreas"):
        pancreas_bbox(np.zeros((8, 8, 8), dtype=np.uint8))


def test_accepts_a_channel_first_mask():
    assert pancreas_bbox(make_mask()[None], margin_mm=0.0, spacing_mm=SP) == \
        pancreas_bbox(make_mask(), margin_mm=0.0, spacing_mm=SP)


def test_crop_has_the_requested_shape_and_keeps_the_organ():
    rng = np.random.default_rng(0)
    image = rng.random((40, 40, 40)).astype(np.float32)
    mask = make_mask()
    roi = crop_roi(image, mask, out_size=(24, 24, 24), margin_mm=2.0, spacing_mm=SP)
    assert roi["image"].shape == roi["mask"].shape == (24, 24, 24)
    assert roi["image"].dtype == np.float32
    assert (roi["mask"] > 0).sum() == (mask > 0).sum()      # nothing of the organ was cut off
    assert roi["spacing_mm"] == SP


def test_crop_values_come_from_the_source_without_resampling():
    image = np.arange(40 ** 3, dtype=np.float32).reshape(40, 40, 40)
    roi = crop_roi(image, make_mask(), out_size=(10, 6, 10), margin_mm=0.0, spacing_mm=SP)
    # window == the tight box here, so the crop must equal that slice exactly
    assert np.array_equal(roi["image"], image[10:20, 12:18, 15:25])


def test_window_off_the_volume_edge_is_zero_padded():
    image = np.ones((40, 40, 40), dtype=np.float32)
    mask = make_mask(box=((0, 6), (0, 6), (0, 6)))
    roi = crop_roi(image, mask, out_size=(20, 20, 20), margin_mm=0.0, spacing_mm=SP)
    assert roi["image"].shape == (20, 20, 20)
    assert roi["image"].min() == 0.0 and roi["image"].max() == 1.0   # padding present, data intact
    assert (roi["mask"] > 0).sum() == 6 ** 3


def test_oversized_organ_is_truncated_not_resized():
    mask = make_mask(box=((5, 35), (5, 35), (5, 35)))
    roi = crop_roi(np.ones((40, 40, 40), np.float32), mask, out_size=(16, 16, 16),
                   margin_mm=0.0, spacing_mm=SP)
    assert roi["image"].shape == (16, 16, 16)
    assert not bbox_fits(roi["bbox"], (16, 16, 16))
    assert bbox_fits(roi["bbox"], (30, 30, 30))


def test_channel_axis_is_carried_through():
    image = np.random.default_rng(1).random((1, 40, 40, 40)).astype(np.float32)
    roi = crop_roi(image, make_mask(), out_size=(16, 16, 16), margin_mm=0.0, spacing_mm=SP)
    assert roi["image"].shape == (1, 16, 16, 16)
    assert roi["mask"].shape == (16, 16, 16)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="spatial shape"):
        crop_roi(np.zeros((30, 30, 30)), make_mask(), out_size=(8, 8, 8))
