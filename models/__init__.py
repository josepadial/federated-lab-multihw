"""
Unified model factory and defaults.

make_model(name, num_classes, in_ch=3, **cfg) -> nn.Module
get_default_input_size(name) -> tuple
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .cnn import CNN
from .efficientnet_lite0 import EfficientNetLite0
from .mlp import MLP
from .mobilenetv3 import MobileNetV3


def make_model(name: str, num_classes: int, in_ch: int = 3, **cfg) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return CNN(in_ch=in_ch, num_classes=num_classes, **cfg)
    if name == "mlp":
        input_size = cfg.pop("input_size", 32 * 32 * in_ch)
        return MLP(input_size=input_size, num_classes=num_classes, **cfg)
    if name == "mobilenetv3":
        # Our wrapper accepts only num_classes; torchvision assumes 3-channel input
        return MobileNetV3(num_classes=num_classes, **cfg)
    if name in ("efficientnet_lite0", "efficientnetlite0"):
        return EfficientNetLite0(num_classes=num_classes, **cfg)
    raise ValueError(f"Unknown model name: {name}")


def get_default_input_size(name: str) -> Tuple[int, ...]:
    n = name.lower()
    if n == "mlp":
        # Default to CIFAR-10 flattened
        return (1, 32 * 32 * 3)
    # Vision backbones default to CIFAR-10 shape
    return (1, 3, 32, 32)
