"""
cnn.py

Configurable CNN for classification with production-friendly defaults.

- NCHW input.
- Conv-BN-Activation blocks (no in-place), optional depthwise and residual.
- Common head: GAP -> Dropout -> Linear(num_classes).
- Kaiming init and ONNX export helper.
"""

from typing import Optional, List

import torch
import torch.nn as nn


def _kaiming_init(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Apply Kaiming initialization to Conv/Linear and zero biases."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: str = "relu", bn_momentum: float = 0.1):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        if act == "relu":
            self.act = nn.ReLU(inplace=False)
            self._nonlin = "relu"
        elif act == "silu":
            self.act = nn.SiLU(inplace=False)
            self._nonlin = "silu"
        else:
            raise ValueError(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, ch: int, dw: bool = False, act: str = "relu", bn_momentum: float = 0.1):
        super().__init__()
        if dw:
            self.block = nn.Sequential(
                ConvBNAct(ch, ch, k=3, s=1, groups=ch, act=act, bn_momentum=bn_momentum),
                ConvBNAct(ch, ch, k=1, s=1, act=act, bn_momentum=bn_momentum),
            )
        else:
            self.block = nn.Sequential(
                ConvBNAct(ch, ch, k=3, s=1, act=act, bn_momentum=bn_momentum),
                ConvBNAct(ch, ch, k=3, s=1, act=act, bn_momentum=bn_momentum),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """Configurable CNN baseline.

    Args:
        in_ch: input channels.
        num_classes: classification classes.
        input_size: assumed H=W for computing head.
        width_mult: channel multiplier.
        dropout: dropout probability before classifier.
        depthwise: whether to use depthwise separable conv in blocks.
        residual: whether to add simple residual connections.
        act: 'relu' or 'silu'.
        bn_momentum: BatchNorm momentum.
    """

    def __init__(self,
                 in_ch: int = 3,
                 num_classes: int = 10,
                 input_size: int = 32,
                 width_mult: float = 1.0,
                 dropout: float = 0.2,
                 depthwise: bool = False,
                 residual: bool = False,
                 act: str = "relu",
                 bn_momentum: float = 0.1):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        layers: List[nn.Module] = [
            ConvBNAct(in_ch, c1, k=3, s=1, act=act, bn_momentum=bn_momentum),
            nn.MaxPool2d(2),
            ConvBNAct(c1, c2, k=3, s=1, act=act, bn_momentum=bn_momentum),
            nn.MaxPool2d(2),
        ]
        block = BasicBlock(c2, dw=depthwise, act=act, bn_momentum=bn_momentum)
        if residual:
            self.features = nn.Sequential(*layers, nn.Sequential(block), )
            self.residual = True
            self.residual_block = block
        else:
            self.features = nn.Sequential(*layers, block)
            self.residual = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(c2, num_classes)
        _kaiming_init(self, nonlinearity=act if act in ("relu", "silu") else "relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.residual:
            x = x + self.residual_block(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def to_onnx(self, sample_shape, out_path, opset: int = 17, dynamic_batch: bool = False) -> None:
        """Export model to ONNX with stable settings."""
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


if __name__ == "__main__":
    m = CNN(in_ch=3, num_classes=10)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"CNN params: {n_params}")
    onnx_path = "models_saved/onnx/test/cnn_test.onnx"
    import os

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    m.to_onnx((1, 3, 32, 32), onnx_path, opset=17, dynamic_batch=True)
    print("Exported:", onnx_path)
