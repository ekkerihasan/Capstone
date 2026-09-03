"""Networks for the pipeline. Stage 6.2 is the Attention U-Net; the classifier
for 6.3 lands here later.

Attention U-Net (Oktay et al., MIDL 2018) is the approved target model: its
attention gates suppress irrelevant background before the skip connections are
concatenated, which is the whole point when the pancreas is ~1% of the volume.
"""
from __future__ import annotations

import torch
from monai.networks.nets import AttentionUnet

from .config import LABEL_TUMOR

N_CLASSES = LABEL_TUMOR + 1          # background / pancreas / tumor

# Fits 96^3 patches at batch size 2 inside 8 GB with AMP on.
CHANNELS_8GB = (16, 32, 64, 128, 256)
STRIDES_8GB = (2, 2, 2, 2)


def build_attention_unet(
    in_channels: int = 1,
    out_channels: int = N_CLASSES,
    channels=CHANNELS_8GB,
    strides=STRIDES_8GB,
    dropout: float = 0.0,
    spatial_dims: int = 3,
) -> AttentionUnet:
    """The stage 6.2 segmentation network.

    `dropout` is left at 0 for training; stage 6.4 turns it on at inference for
    MC Dropout, so the layers have to exist in the checkpoint from the start if
    that variant is used — keep it explicit in the run config rather than
    flipping it silently later.
    """
    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"strides must have len(channels) - 1 entries, got {len(strides)} for {len(channels)} channels"
        )
    return AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        dropout=dropout,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: torch.nn.Module) -> str:
    n = count_parameters(model)
    return f"{type(model).__name__}: {n:,} trainable parameters ({n * 4 / 1e6:.1f} MB fp32)"
