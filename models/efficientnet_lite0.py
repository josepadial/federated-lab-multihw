"""
efficientnet_lite0.py

Implementation of EfficientNet-Lite0 for image classification (e.g., CIFAR-10).
Includes model definition and utility to load trained weights.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


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

    def to_onnx(self, sample_shape, out_path, opset: int = 17, dynamic_batch: bool = False) -> None:
        self.eval()
        dummy = torch.randn(*sample_shape)
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
        torch.onnx.export(
            self,
            dummy,
            out_path,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
