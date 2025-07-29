"""
export_utils.py

Utilities for saving, loading, exporting, and quantizing PyTorch models for edge/federated learning.
"""

import os
from typing import Optional, Dict, Any

import torch
import torch.onnx
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Base folder to store models
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')


def save_model(
        model: nn.Module,
        file_name: str,
        optimizer: Optional[Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves a PyTorch model in .pt format in the data_root folder, optionally including the optimizer state and other data.

    Args:
        model (nn.Module): Model to save.
        file_name (str): .pt file name (not full path).
        optimizer (Optimizer, optional): Optimizer to save.
        extra (dict, optional): Dictionary with additional information to save.
    """
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, file_name)
    state = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_model(
        model: nn.Module,
        file_name: str,
        optimizer: Optional[Optimizer] = None,
        map_location: Optional[str] = None
) -> Optional[Optimizer]:
    """
    Loads the weights of a PyTorch model from a .pt file in the data_root folder. Can also load the optimizer state.

    Args:
        model (nn.Module): Model to load weights into.
        file_name (str): .pt file name (not full path).
        optimizer (Optimizer, optional): Optimizer to load its state.
        map_location (str, optional): Target device ('cpu', 'cuda', etc.).

    Returns:
        Optimizer or None: The optimizer with loaded state, if provided.
    """
    path = os.path.join(data_root, file_name)
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer
    return None


def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        path: str,
        num_classes: Optional[int] = None,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        opset_version: int = 12,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> None:
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (nn.Module): Model to export.
        dummy_input (torch.Tensor): Simulated input tensor (shape as expected by the model).
        path (str): Path to save the .onnx file.
        num_classes (int, optional): Number of output classes (if applicable).
        input_names (list, optional): Input names.
        output_names (list, optional): Output names.
        opset_version (int): ONNX opset version.
        dynamic_axes (dict, optional): Dynamic axes for variable batch size, etc.
    """
    model.eval()
    if num_classes is not None and hasattr(model, 'num_classes'):
        model.num_classes = num_classes
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=dynamic_axes or {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )


def quantize_model_dynamic(
        model: nn.Module
) -> nn.Module:
    """
    Applies dynamic quantization (INT8) to a PyTorch model and returns the quantized model.

    Args:
        model (nn.Module): Model to quantize.

    Returns:
        nn.Module: Quantized model (dynamically).
    """
    import torch.quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM, nn.GRU, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model


def quantize_model_static(
        model: nn.Module,
        calibration_loader: DataLoader,
        device: str = 'cpu'
) -> nn.Module:
    """
    Applies static quantization (INT8) to a PyTorch model using a calibration DataLoader.

    Args:
        model (nn.Module): Model to quantize.
        calibration_loader (DataLoader): DataLoader for calibration.
        device (str): Device ('cpu' recommended for static quantization).

    Returns:
        nn.Module: Quantized model (statically).
    """
    import torch.quantization
    model.eval()
    model.to(device)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            model(inputs)
    torch.quantization.convert(model, inplace=True)
    return model


def compare_model_size_and_accuracy(
        model_fp: nn.Module,
        model_int8: nn.Module,
        test_loader: 'DataLoader',
        device: str = 'cpu',
        criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    Compares the size and accuracy between an original model and its quantized version.

    Args:
        model_fp (nn.Module): Original model (float32).
        model_int8 (nn.Module): Quantized model (int8).
        test_loader (DataLoader): Test DataLoader.
        device (str): Device for evaluation.
        criterion (nn.Module, optional): Loss function.

    Returns:
        Dict[str, Any]: Dictionary with sizes (bytes), accuracy, and difference.
    """
    import tempfile
    import os
    # Temporarily save both models
    with tempfile.NamedTemporaryFile(delete=False) as f_fp, tempfile.NamedTemporaryFile(delete=False) as f_int8:
        torch.save(model_fp.state_dict(), f_fp.name)
        torch.save(model_int8.state_dict(), f_int8.name)
        size_fp = os.path.getsize(f_fp.name)
        size_int8 = os.path.getsize(f_int8.name)

    # Evaluate accuracy
    def eval_acc(model):
        model.eval()
        model.to(device)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    acc_fp = eval_acc(model_fp)
    acc_int8 = eval_acc(model_int8)
    # Clean up temporary files
    os.remove(f_fp.name)
    os.remove(f_int8.name)
    return {
        'size_fp32_bytes': size_fp,
        'size_int8_bytes': size_int8,
        'accuracy_fp32': acc_fp,
        'accuracy_int8': acc_int8,
        'accuracy_diff': acc_fp - acc_int8
    }
