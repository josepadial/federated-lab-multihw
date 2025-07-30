"""
cnn.py

Implementation of a simple CNN for image classification (e.g., CIFAR-10).
Includes model definition and utility to load trained weights.
"""

from typing import Optional

import torch
import torch.nn as nn

from utils.export_utils import load_trained_model


class CNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB).
        num_classes (int): Number of output classes.
        input_size (int): Height/width of input images (assumed square).
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10, input_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        # Calculate input size for the linear layer
        fc_input_size = 64 * (input_size // 4) * (input_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_trained_cnn(
        path: Optional[str] = None,
        input_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
        map_location: str = 'cpu'
) -> CNN:
    """
    Instantiates a CNN model and loads trained weights from the specified path or from models_saved/cnn_cifar10.pt.
    """
    return load_trained_model(
        model_class=CNN,
        model_kwargs={'input_channels': input_channels, 'num_classes': num_classes, 'input_size': input_size},
        default_filename='cnn_cifar10.pt',
        path=path,
        map_location=map_location
    )
