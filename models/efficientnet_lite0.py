"""
efficientnet_lite0.py

Implementation of EfficientNet-Lite0 for image classification (e.g., CIFAR-10).
Includes model definition and utility to load trained weights.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from utils.export_utils import load_trained_model


class EfficientNetLite0(nn.Module):
    """
    EfficientNet-Lite0 adapted for small images (e.g., CIFAR-10, 32x32) and custom number of classes.

    Args:
        num_classes (int): Number of output classes (default 10).
        pretrained (bool): Whether to load ImageNet pre-trained weights (default False).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # Load EfficientNet-B0 base
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        # Adjust first conv layer for 32x32 images (stride=1)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # Adjust classifier for custom number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EfficientNet-Lite0.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 32, 32).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        return self.model(x)


def load_trained_efficientnetlite0(
        path: Optional[str] = None,
        num_classes: int = 10,
        pretrained: bool = False,
        map_location: str = 'cpu'
) -> EfficientNetLite0:
    """
    Instantiates an EfficientNetLite0 model and loads trained weights from the specified path or from models_saved/efficientnetlite0_cifar10.pt.
    """
    return load_trained_model(
        model_class=EfficientNetLite0,
        model_kwargs={'num_classes': num_classes, 'pretrained': pretrained},
        default_filename='efficientnetlite0_cifar10.pt',
        path=path,
        map_location=map_location
    )
