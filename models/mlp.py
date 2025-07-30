"""
mlp.py

Implementation of a Multilayer Perceptron (MLP) for image classification (e.g., CIFAR-10, FashionMNIST).
Includes model definition and utility to load trained weights.
"""

from typing import Optional

import torch
import torch.nn as nn

from utils.export_utils import load_trained_model


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for small image classification tasks.

    Architecture:
        Flatten → Linear → ReLU → Linear → ReLU → Linear (output)

    Args:
        input_size (int): Number of input features (default 28*28 for FashionMNIST, 32*32*3 for CIFAR-10).
        num_classes (int): Number of output classes (default 10).
        hidden1 (int): Units in the first hidden layer (default 256).
        hidden2 (int): Units in the second hidden layer (default 128).
    """

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) or (N, input_size).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def load_trained_mlp(
        path: Optional[str] = None,
        input_size: int = 32 * 32 * 3,
        num_classes: int = 10,
        hidden1: int = 256,
        hidden2: int = 128,
        map_location: str = 'cpu'
) -> MLP:
    """
    Instantiates an MLP model and loads trained weights from the specified path or from models_saved/mlp_cifar10.pt.
    """
    return load_trained_model(
        model_class=MLP,
        model_kwargs={'input_size': input_size, 'num_classes': num_classes, 'hidden1': hidden1, 'hidden2': hidden2},
        default_filename='mlp_cifar10.pt',
        path=path,
        map_location=map_location
    )
