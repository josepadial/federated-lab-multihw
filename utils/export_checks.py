"""Helpers to export models to ONNX and run basic checks."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def try_export_onnx(model: torch.nn.Module,
                    out_path: str | Path,
                    sample_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
                    opset: int = 17,
                    dynamic_batch: bool = True) -> Path:
    """Try to export a torch.nn.Module via its to_onnx or generic torch.onnx.export.

    Returns the output path on success, raises on failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    if hasattr(model, "to_onnx"):
        model.to_onnx(sample_shape, str(out_path), opset=opset, dynamic_batch=dynamic_batch)
    else:
        dummy = torch.randn(*sample_shape)
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
        torch.onnx.export(model, dummy, str(out_path), export_params=True, do_constant_folding=True,
                          input_names=["input"], output_names=["logits"], dynamic_axes=dynamic_axes,
                          opset_version=opset)
    return out_path
