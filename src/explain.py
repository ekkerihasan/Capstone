"""Stage 6.5, part one: Grad-CAM++ saliency.

Grad-CAM++ (Chattopadhay et al., WACV 2018) weights each feature map by how
much *positively* it contributes to the class score, rather than by the plain
gradient average Grad-CAM uses. For a target class c and the feature maps A^k
of a chosen layer:

    alpha^k = grad^2 / (2 * grad^2 + sum_spatial(A^k * grad^3))
    w^k     = sum_spatial(alpha^k * relu(grad))
    CAM     = relu(sum_k w^k * A^k)

The CAM is then resized to the input and normalised to [0, 1].

Written out rather than taken from `pytorch-grad-cam` for two reasons: this
pipeline is 3D throughout and the library's 3D support is awkward, and the
method has to be defended at the viva, which is easier when it is forty lines
in the repo than a call into a dependency.

A saliency map is a claim about the model, and an unverified one is worth
nothing - every map this module produces should go through `src/fidelity.py`
before it is shown to anybody.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def find_target_layer(model: nn.Module) -> nn.Module:
    """The last convolution in the network.

    The convention for Grad-CAM: the deepest convolutional layer still carries
    spatial structure while holding the most semantic information. Anything
    after it has usually been pooled flat.
    """
    convs = [m for m in model.modules() if isinstance(m, CONV_TYPES)]
    if not convs:
        raise ValueError("no convolution found - pass target_layer explicitly")
    return convs[-1]


class GradCAMPlusPlus:
    """Hooks a layer, runs one forward and one backward pass, returns the CAM.

    Use it as a context manager, or call `remove()`: the hooks hold references
    to activations and would otherwise keep a volume's worth of tensors alive
    after the explanation is done.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.layer = target_layer if target_layer is not None else find_target_layer(model)
        self._activations = None
        self._gradients = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activation),
            self.layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self._activations = out

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x, class_idx: int | None = None, normalise: bool = True):
        """-> (cam, class_idx, score). `cam` has the spatial shape of `x`."""
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = torch.as_tensor(x).clone().requires_grad_(True)
        logits = self.model(x)
        if logits.ndim != 2:
            raise ValueError(
                f"expected classifier logits [N, C], got {tuple(logits.shape)}. "
                "Wrap a segmentation model in ScoreHead first.")
        if class_idx is None:
            class_idx = int(logits[0].argmax())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self._activations[0].detach()        # [K, *spatial]
        grads = self._gradients[0].detach()         # [K, *spatial]
        spatial = tuple(range(1, acts.ndim))

        g2, g3 = grads.pow(2), grads.pow(3)
        denom = 2.0 * g2 + (acts * g3).sum(dim=spatial, keepdim=True)
        alpha = g2 / torch.where(denom != 0.0, denom, torch.ones_like(denom))
        weights = (alpha * F.relu(grads)).sum(dim=spatial)               # [K]

        cam = F.relu((weights.view(-1, *([1] * len(spatial))) * acts).sum(0))
        cam = F.interpolate(cam[None, None], size=tuple(x.shape[2:]),
                            mode="trilinear" if cam.ndim == 3 else "bilinear",
                            align_corners=False)[0, 0]

        cam = cam.cpu().numpy()
        if normalise:
            lo, hi = float(cam.min()), float(cam.max())
            cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)

        if was_training:
            self.model.train()
        return cam, int(class_idx), float(score.detach())


def gradcam_plusplus(model: nn.Module, x, target_layer=None, class_idx=None):
    """One-shot convenience wrapper; hooks are always removed."""
    with GradCAMPlusPlus(model, target_layer) as cam:
        return cam(x, class_idx=class_idx)


class ScoreHead(nn.Module):
    """Turn a segmentation model into a two-class scorer.

    Grad-CAM++ needs a scalar class score to differentiate, and stage 6.3's
    classifier does not exist yet. This reduces the segmentation logits to
    [background, tumor] by averaging the top-k voxels of each channel, which is
    differentiable and keeps the whole graph intact.

    It is a stand-in, not the real thing: when `classifier.py` lands, Grad-CAM++
    is pointed at that model instead and this class is only needed if anyone
    wants to explain the segmentation network itself.
    """

    def __init__(self, seg_model: nn.Module, tumor_channel: int = 2,
                 background_channel: int = 0, top_k: int = 32, divisor: int = 16):
        super().__init__()
        self.seg = seg_model
        self.tumor_channel = tumor_channel
        self.background_channel = background_channel
        self.top_k = top_k
        self.divisor = divisor      # U-Net depth: 4 downsamples -> 16

    def _pad(self, x):
        """Pad up to a multiple of `divisor`.

        The U-Net halves its spatial dims four times and adds the skip
        connections back, so an input of 26x23x50 fails on the concatenation.
        Validation gets around this with a sliding window, but Grad-CAM++ needs
        one differentiable forward pass over the whole volume, so it pads
        instead. The padding is 0.0, the bottom of the stage 6.1 HU window.
        """
        pads, shape = [], x.shape[2:]
        for s in reversed(shape):
            need = (-s) % self.divisor
            pads.extend([0, need])
        return F.pad(x, pads) if any(pads) else x

    def _pool(self, logits, channel):
        flat = logits[:, channel].flatten(1)
        k = min(self.top_k, flat.shape[1])
        return flat.topk(k, dim=1).values.mean(dim=1)

    def forward(self, x):
        logits = self.seg(self._pad(x))
        return torch.stack([self._pool(logits, self.background_channel),
                            self._pool(logits, self.tumor_channel)], dim=1)


def overlay_saliency(image, saliency, path=None, title: str = "", gt=None,
                     stamp: bool = True, z: int | None = None,
                     top_k_frac: float = 0.10):
    """CT, the heat map, and the region the fidelity test actually masks.

    The third panel is not a translucent wash of the whole heat map. It outlines
    exactly the top `top_k_frac` of voxels - the ones `fidelity.mask_top_k`
    hides when it tests the explanation - so the picture and the PASS/FAIL
    verdict refer to the same region rather than to two loosely related things.

    The heat map panel is stretched between its 50th and 99.5th percentile. A
    Grad-CAM map from a weakly-trained network sits bunched near the top of
    [0, 1]; drawn against a fixed 0-1 scale it is a solid red rectangle, which
    looks like a bug and shows nothing.
    """
    import matplotlib.pyplot as plt

    from .config import LABEL_TUMOR
    from .phantom import stamp_figure
    from .viz import save_fig

    img = np.asarray(image).squeeze()
    sal = np.asarray(saliency).squeeze()
    if z is None:
        # Prefer the slice holding the most tumour: the question this figure
        # answers is "did it look where the tumour is", which cannot be read off
        # a slice the tumour is not on. Only when there is no tumour to show
        # does the brightest saliency slice become the useful one.
        z = None
        if gt is not None:
            tumour = np.asarray(gt).squeeze() == LABEL_TUMOR
            if tumour.any():
                z = int(tumour.sum(axis=(0, 1)).argmax())
        if z is None:
            z = int(sal.sum(axis=(0, 1)).argmax()) if sal.any() else img.shape[2] // 2

    # the top-k voxels over the WHOLE volume, so the slice shows this slice's
    # share of what the test masks rather than its own local maximum
    flat = sal.ravel()
    n = max(1, int(round(top_k_frac * flat.size)))
    cut = float(np.partition(flat, -n)[-n]) if flat.size else 1.0

    im = np.rot90(img[:, :, z])
    sl = np.rot90(sal[:, :, z])
    hot = sl >= cut

    lo, hi = np.percentile(sal, 50), np.percentile(sal, 99.5)
    if hi <= lo:
        lo, hi = float(sal.min()), max(float(sal.max()), float(sal.min()) + 1e-6)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
    ax[0].imshow(im, cmap="gray")
    ax[0].set_title("preprocessed CT", fontsize=11)

    hm = ax[1].imshow(sl, cmap="inferno", vmin=lo, vmax=hi)
    ax[1].set_title("where the model looked", fontsize=11)
    cb = fig.colorbar(hm, ax=ax[1], fraction=0.046)
    cb.set_ticks([lo, hi])
    cb.set_ticklabels(["barely", "hardest"])

    ax[2].imshow(im, cmap="gray")
    ax[2].imshow(np.ma.masked_where(~hot, hot.astype(float)), cmap="autumn", alpha=0.55)
    ax[2].contour(hot, colors="#ff3b30", linewidths=1.2)
    legend = f"red = top {top_k_frac:.0%} the test hides"
    if gt is not None:
        ax[2].contour(np.rot90(np.asarray(gt).squeeze()[:, :, z]) == LABEL_TUMOR,
                      colors="white", linewidths=1.4)
        legend += " · white = true tumor"
    ax[2].set_title(legend, fontsize=10)

    for a in ax:
        a.axis("off")
    fig.suptitle(f"{title}  |  axial slice z={z}", fontsize=11)
    fig.tight_layout()
    if stamp:
        stamp_figure(fig)
    if path:
        save_fig(fig, path)
        plt.close(fig)
    return fig
