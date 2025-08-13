"""
mlp.py

Production-friendly MLP for image classification.

- Accepts (N, C, H, W) or (N, F) input. Flattens internally.
- LayerNorm + GELU + Dropout on hidden layers.
- Xavier init and ONNX export helper.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer Perceptron with LN+GELU+Dropout layers.

    Args:
        input_size: flattened input features (default 32*32*3 for CIFAR-10).
        num_classes: output classes.
        hidden1: first hidden size.
        hidden2: second hidden size.
        dropout: dropout probability.
    """

    def __init__(self, input_size: int = 32 * 32 * 3, num_classes: int = 10, hidden1: int = 256, hidden2: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.ln_in = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden2, num_classes)
        # Xavier init for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) or (N, input_size).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        if x.dim() > 2:
            x = self.flatten(x)
        x = self.ln_in(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

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
