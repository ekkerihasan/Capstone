"""Stage 6.3, part two: the tumor present/absent classifier.

A 3D DenseNet-121 or ResNet from MONAI, run on the ROI crop from `roi.py`.
Output is two logits per case; softmax of the second is `prob_tumor`.

`dropout` exists for stage 6.4. MC Dropout needs Dropout modules to switch on at
inference, so they have to be in the network from construction. MONAI's DenseNet
takes `dropout_prob` natively; its ResNets have no dropout, so one is placed in
front of the final linear layer here. Keep it at 0.0 for plain training.

Minimum ROI edge is 32 voxels for DenseNet-121 (five stride-2 stages); the ResNets
cope with smaller inputs, which is why the unit tests use 16.

Signature frozen in docs/REVIEW2_TEMP_PLAN.md section 9.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet10, resnet18, resnet34

N_CLASSES_CLS = 2                  # tumor absent / present

_RESNETS = {"resnet10": resnet10, "resnet18": resnet18, "resnet34": resnet34}
AVAILABLE = ("densenet121",) + tuple(_RESNETS)


def build_classifier(name: str = "densenet121", in_channels: int = 1,
                     num_classes: int = N_CLASSES_CLS, dropout: float = 0.0) -> nn.Module:
    name = name.lower()
    if name == "densenet121":
        return DenseNet121(spatial_dims=3, in_channels=in_channels,
                           out_channels=num_classes, dropout_prob=dropout)
    if name in _RESNETS:
        model = _RESNETS[name](spatial_dims=3, n_input_channels=in_channels,
                               num_classes=num_classes)
        if dropout > 0.0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), model.fc)
        return model
    raise ValueError(f"unknown classifier '{name}'; choose from {AVAILABLE}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    return f"{type(model).__name__}: {count_parameters(model) / 1e6:.2f} M trainable parameters"


@torch.no_grad()
def tumor_probability(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Softmax P(tumor present) for a batch, shape [N]."""
    model.eval()
    return torch.softmax(model(x), dim=1)[:, 1]
