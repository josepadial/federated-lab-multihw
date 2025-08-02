"""
export_utils.py

Utilities for saving, loading, exporting, and quantizing PyTorch models for edge/federated learning.
"""

import logging
import os
import tempfile
import time
from typing import Optional, Dict, Any, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx
import torch.quantization
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base folder to store models
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')

MODEL_INSTANCE_ERROR = "model must be a torch.nn.Module instance."
PROVIDED_MODEL_INSTANCE_ERROR = "Provided model is not a torch.nn.Module instance."


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

    Raises:
        ValueError: If model is not an nn.Module or file_name is not a string.
    """
    if not isinstance(model, nn.Module):
        logger.error(PROVIDED_MODEL_INSTANCE_ERROR)
        raise ValueError(MODEL_INSTANCE_ERROR)
    if not isinstance(file_name, str):
        logger.error("file_name must be a string.")
        raise ValueError("file_name must be a string.")
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, file_name)
    state = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)
    logger.info(f"Model saved to {path}")


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

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If model is not an nn.Module.
    """
    if not isinstance(model, nn.Module):
        logger.error(PROVIDED_MODEL_INSTANCE_ERROR)
        raise ValueError(MODEL_INSTANCE_ERROR)
    path = os.path.join(data_root, file_name)
    if not os.path.isfile(path):
        logger.error(f"Model file {path} does not exist.")
        raise FileNotFoundError(f"Model file {path} does not exist.")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from {path}")
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded.")
        return optimizer
    return None


def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        path: str,
        num_classes: Optional[int] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
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
        dynamic_axes (dict, optional): Dynamic axes for variable batch size, etc

    Raises:
        ValueError: If model or dummy_input are not valid.
    """
    if not isinstance(model, nn.Module):
        logger.error(PROVIDED_MODEL_INSTANCE_ERROR)
        raise ValueError(MODEL_INSTANCE_ERROR)
    if not isinstance(dummy_input, torch.Tensor):
        logger.error("dummy_input must be a torch.Tensor.")
        raise ValueError("dummy_input must be a torch.Tensor.")
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
    logger.info(f"Model exported to ONNX at {path}")


def quantize_model_dynamic(
        model: nn.Module
) -> nn.Module:
    """
    Applies dynamic quantization (INT8) to a PyTorch model and returns the quantized model.

    Args:
        model (nn.Module): Model to quantize.

    Returns:
        nn.Module: Quantized model (dynamically).

    Raises:
        ValueError: If model is not an nn.Module.
    """
    if not isinstance(model, nn.Module):
        logger.error(PROVIDED_MODEL_INSTANCE_ERROR)
        raise ValueError(MODEL_INSTANCE_ERROR)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM, nn.GRU, nn.Conv2d}, dtype=torch.qint8
    )
    logger.info("Model dynamically quantized (INT8).")
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

    Raises:
        ValueError: If model is not an nn.Module or calibration_loader is not a DataLoader.
    """
    if not isinstance(model, nn.Module):
        logger.error(PROVIDED_MODEL_INSTANCE_ERROR)
        raise ValueError(MODEL_INSTANCE_ERROR)
    if not isinstance(calibration_loader, DataLoader):
        logger.error("calibration_loader must be a torch.utils.data.DataLoader.")
        raise ValueError("calibration_loader must be a torch.utils.data.DataLoader.")

    model.eval()
    model.to(device)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            model(inputs)
    torch.quantization.convert(model, inplace=True)
    logger.info("Model statically quantized (INT8).")
    return model


def compare_model_size_and_accuracy(
        model_fp: nn.Module,
        model_int8: nn.Module,
        test_loader: DataLoader,
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

    # Temporarily save both models
    with tempfile.NamedTemporaryFile(delete=False) as f_fp, tempfile.NamedTemporaryFile(delete=False) as f_int8:
        torch.save(model_fp.state_dict(), f_fp.name)
        torch.save(model_int8.state_dict(), f_int8.name)
        size_fp = os.path.getsize(f_fp.name)
        size_int8 = os.path.getsize(f_int8.name)

    def eval_acc(model: nn.Module) -> float:
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
    os.remove(f_fp.name)
    os.remove(f_int8.name)
    logger.info(f"Model size (FP32): {size_fp} bytes, Model size (INT8): {size_int8} bytes")
    logger.info(f"Accuracy (FP32): {acc_fp:.4f}, Accuracy (INT8): {acc_int8:.4f}")
    return {
        'size_fp32_bytes': size_fp,
        'size_int8_bytes': size_int8,
        'accuracy_fp32': acc_fp,
        'accuracy_int8': acc_int8,
        'accuracy_diff': acc_fp - acc_int8
    }


def load_trained_model(
        model_class: type,
        model_kwargs: dict,
        default_filename: str,
        path: Optional[str] = None,
        map_location: str = 'cpu'
) -> nn.Module:
    """
    Instantiates a model and loads trained weights from the specified path or a default location.

    Args:
        model_class (type): The class of the model to instantiate.
        model_kwargs (dict): Arguments to pass to the model constructor.
        default_filename (str): Default filename for the weights.
        path (str, optional): Path to the .pt file. If None, uses default location.
        map_location (str): Device to map the model ('cpu', 'cuda', etc).

    Returns:
        nn.Module: Model with loaded weights, set to eval mode.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If loading the state dict fails.
    """
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')
    if path is None:
        path = os.path.join(data_root, default_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model weights file not found: {path}")
    model = model_class(**model_kwargs)
    state = torch.load(path, map_location=map_location, weights_only=True)
    try:
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Failed to load model state dict: {e}")
    model.eval()
    return model


def check_onnx_integrity(path: str) -> bool:
    """Checks if the ONNX model at path is well-formed."""
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return True


def compare_pytorch_onnx_outputs(pytorch_model, onnx_path, test_batch):
    """Compares outputs between PyTorch and ONNX Runtime for a batch."""
    pytorch_model.eval()
    with torch.no_grad():
        torch_out = pytorch_model(test_batch).cpu().numpy()
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_batch.cpu().numpy()}
    onnx_out = ort_session.run(None, ort_inputs)[0]
    max_diff = np.max(np.abs(torch_out - onnx_out))
    return max_diff, torch_out, onnx_out


def verify_onnx_dynamic_batch(onnx_path, input_shape, batch_sizes=[1, 4, 16]):
    """Verifies ONNX model supports dynamic batch sizes."""
    ort_session = ort.InferenceSession(str(onnx_path))
    results = []
    for batch_size in batch_sizes:
        dyn_input = torch.randn(batch_size, *input_shape).cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: dyn_input}
        dyn_out = ort_session.run(None, ort_inputs)[0]
        results.append({'Batch Size': batch_size, 'Output Shape': dyn_out.shape})
    return results


def benchmark_onnx_inference(onnx_path, input_array, num_runs=100):
    """Benchmarks ONNX Runtime inference latency."""
    ort_session = ort.InferenceSession(str(onnx_path))
    start = time.time()
    for _ in range(num_runs):
        ort_session.run(None, {ort_session.get_inputs()[0].name: input_array})
    avg_time = (time.time() - start) / num_runs
    return avg_time


def compare_model_file_sizes(pytorch_path, onnx_path):
    """Returns file sizes in KB for PyTorch and ONNX models."""
    pytorch_size = os.path.getsize(pytorch_path) / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024
    return pytorch_size, onnx_size
