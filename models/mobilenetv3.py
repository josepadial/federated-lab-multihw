"""
mobilenetv3.py

Implementation of MobileNetV3 Small for image classification (e.g., CIFAR-10).
Includes model definition and utility to load trained weights.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from utils.export_utils import load_trained_model


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Small adapted for small image classification tasks (e.g., CIFAR-10, 32x32).

    Utilizes torchvision's MobileNetV3 Small, adjusting the first layer and final classifier
    for 32x32 images and a custom number of classes.

    Args:
        num_classes (int): Number of output classes (default 10).
        weights (Optional): Weights to use (default None for random initialization).
    """

    def __init__(self, num_classes: int = 10, weights: Optional[MobileNet_V3_Small_Weights] = None):
        super().__init__()
        self.model = mobilenet_v3_small(weights=weights)
        # Adjust first conv layer for 32x32 images (stride=1)
        self.model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # Adjust classifier for custom number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MobileNetV3 Small.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 32, 32).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        return self.model(x)


def load_trained_mobilenetv3(
        path: Optional[str] = None,
        num_classes: int = 10,
        weights: Optional[MobileNet_V3_Small_Weights] = None,
        map_location: str = 'cpu'
) -> 'MobileNetV3':
    """
    Instantiates a MobileNetV3 model and loads trained weights from the specified path or from models_saved/mobilenetv3_cifar10.pt.

    Args:
        path (str, optional): Path to the .pt file. If None, uses default location.
        num_classes (int): Number of output classes.
        weights (Optional): Weights to use for base model.
        map_location (str): Device to map the model ('cpu', 'cuda', etc).
    Returns:
        MobileNetV3: Model with loaded weights, set to eval mode.
    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If loading the state dict fails.
    """
    return load_trained_model(
        model_class=MobileNetV3,
        model_kwargs={'num_classes': num_classes, 'weights': weights},
        default_filename='mobilenetv3_cifar10.pt',
        path=path,
        map_location=map_location
    )
