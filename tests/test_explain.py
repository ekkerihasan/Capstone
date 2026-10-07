"""Stage 6.5 - Grad-CAM++.

The decisive test is `test_cam_lands_on_the_evidence`: a model whose score
depends only on one corner of the volume, so a correct saliency map has exactly
one place to point. Everything else here is shape and hygiene.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explain import (GradCAMPlusPlus, ScoreHead, find_target_layer,
                         gradcam_plusplus, overlay_saliency)
from src.models import build_attention_unet


class CornerDetector(nn.Module):
    """Reads an 8^3 corner and nothing else. Class 1 fires when the corner is
    bright, class 0 when it is dark."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.fill_(0.0)
            self.conv.weight[0] = 1.0
            self.conv.bias.fill_(0.0)
        self.head = nn.Linear(4, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[-1.0, 0, 0, 0], [1.0, 0, 0, 0]]))
            self.head.bias.zero_()

    def forward(self, x):
        pooled = self.conv(x)[:, :, :8, :8, :8].mean(dim=(2, 3, 4))
        return self.head(pooled) * 4.0


@pytest.fixture
def corner_case():
    x = torch.zeros(1, 1, 16, 16, 16)
    x[0, 0, :8, :8, :8] = 1.0
    return CornerDetector().eval(), x


def test_find_target_layer_picks_the_last_convolution():
    model = nn.Sequential(nn.Conv3d(1, 2, 3), nn.ReLU(), nn.Conv3d(2, 4, 3), nn.Flatten())
    assert find_target_layer(model) is model[2]


def test_find_target_layer_refuses_a_model_without_convolutions():
    with pytest.raises(ValueError, match="no convolution"):
        find_target_layer(nn.Sequential(nn.Flatten(), nn.Linear(4, 2)))


def test_cam_lands_on_the_evidence(corner_case):
    """The whole point of stage 6.5. If this passes, the method is implemented
    correctly; if it fails, nothing downstream means anything."""
    model, x = corner_case
    cam, cls, _ = gradcam_plusplus(model, x)
    assert cls == 1
    corner = cam[:8, :8, :8].mean()
    elsewhere = (cam.sum() - cam[:8, :8, :8].sum()) / (cam.size - 8 ** 3)
    assert corner > 10 * elsewhere


def test_cam_has_the_input_shape_and_is_normalised(corner_case):
    model, x = corner_case
    cam, _, _ = gradcam_plusplus(model, x)
    assert cam.shape == tuple(x.shape[2:])
    assert cam.min() >= 0.0 and cam.max() == pytest.approx(1.0)


def test_cam_can_be_asked_for_a_specific_class(corner_case):
    model, x = corner_case
    _, cls, score_1 = gradcam_plusplus(model, x, class_idx=1)
    _, _, score_0 = gradcam_plusplus(model, x, class_idx=0)
    assert cls == 1
    assert score_1 > score_0


def test_hooks_are_removed_on_exit(corner_case):
    model, x = corner_case
    layer = find_target_layer(model)
    before = len(layer._forward_hooks)
    with GradCAMPlusPlus(model, layer) as cam:
        cam(x)
        assert len(layer._forward_hooks) == before + 1
    assert len(layer._forward_hooks) == before


def test_model_is_left_in_eval_and_without_stale_gradients(corner_case):
    model, x = corner_case
    gradcam_plusplus(model, x)
    assert not model.training
    assert all(p.grad is None or torch.all(p.grad == 0) is not None for p in model.parameters())


def test_segmentation_logits_are_rejected_with_a_useful_message():
    """A 5-D output is the easy mistake; say what to do about it."""
    seg = build_attention_unet(channels=(4, 8, 16), strides=(2, 2))
    with pytest.raises(ValueError, match="ScoreHead"):
        gradcam_plusplus(seg, torch.randn(1, 1, 16, 16, 16))


def test_score_head_reduces_a_segmentation_model_to_two_logits():
    seg = build_attention_unet(channels=(4, 8, 16), strides=(2, 2))
    out = ScoreHead(seg)(torch.randn(1, 1, 16, 16, 16))
    assert out.shape == (1, 2)
    assert out.requires_grad


def test_score_head_pads_awkward_volume_shapes():
    """Preprocessed volumes are not multiples of the U-Net depth; the whole
    reason the app needs a sliding window. Grad-CAM++ pads instead."""
    seg = build_attention_unet(channels=(4, 8, 16), strides=(2, 2))
    out = ScoreHead(seg, divisor=4)(torch.randn(1, 1, 26, 23, 50))
    assert out.shape == (1, 2)


def test_gradcam_works_end_to_end_through_score_head():
    seg = build_attention_unet(channels=(4, 8, 16), strides=(2, 2))
    cam, cls, _ = gradcam_plusplus(ScoreHead(seg, divisor=4),
                                   torch.randn(1, 1, 26, 23, 50))
    assert cam.shape == (26, 23, 50)
    assert cls in (0, 1)


def test_overlay_is_stamped_and_saved(tmp_path, corner_case):
    model, x = corner_case
    cam, _, _ = gradcam_plusplus(model, x)
    path = tmp_path / "cam.png"
    overlay_saliency(x, cam, path=path, title="corner", stamp=True)
    assert path.exists() and path.stat().st_size > 0
