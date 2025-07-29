"""
export_utils.py

Utilidades para guardar, cargar, exportar y quantizar modelos PyTorch para edge/federated learning.
"""

import os
from typing import Optional, Dict, Any

import torch
import torch.onnx
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Carpeta base para almacenar los datos
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')


def save_model(
        model: nn.Module,
        file_name: str,
        optimizer: Optional[Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Guarda un modelo PyTorch en formato .pt en la carpeta data_root, incluyendo opcionalmente el estado del optimizador y otros datos.

    Args:
        model (nn.Module): Modelo a guardar.
        file_name (str): Nombre del archivo .pt (no path completo).
        optimizer (Optimizer, opcional): Optimizador a guardar.
        extra (dict, opcional): Diccionario con información adicional a guardar.
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
    Carga los pesos de un modelo PyTorch desde un archivo .pt en la carpeta data_root. Puede cargar también el estado del optimizador.

    Args:
        model (nn.Module): Modelo donde cargar los pesos.
        file_name (str): Nombre del archivo .pt (no path completo).
        optimizer (Optimizer, opcional): Optimizador para cargar su estado.
        map_location (str, opcional): Dispositivo destino ('cpu', 'cuda', etc.).

    Returns:
        Optimizer o None: El optimizador con el estado cargado, si se proporciona.
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
    Exporta un modelo PyTorch a formato ONNX.

    Args:
        model (nn.Module): Modelo a exportar.
        dummy_input (torch.Tensor): Tensor de entrada simulado (shape igual a la esperada por el modelo).
        path (str): Ruta donde guardar el archivo .onnx.
        num_classes (int, opcional): Número de clases de salida (si aplica).
        input_names (list, opcional): Nombres de las entradas.
        output_names (list, opcional): Nombres de las salidas.
        opset_version (int): Versión de ONNX opset.
        dynamic_axes (dict, opcional): Ejes dinámicos para batch size variable, etc.
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
    Aplica quantization dinámica (INT8) a un modelo PyTorch y devuelve el modelo quantizado.

    Args:
        model (nn.Module): Modelo a quantizar.

    Returns:
        nn.Module: Modelo quantizado (dinámicamente).
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
    Aplica quantization estática (INT8) a un modelo PyTorch usando un DataLoader de calibración.

    Args:
        model (nn.Module): Modelo a quantizar.
        calibration_loader (DataLoader): DataLoader para calibración.
        device (str): Dispositivo ('cpu' recomendado para quantization estática).

    Returns:
        nn.Module: Modelo quantizado (estáticamente).
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
    Compara el tamaño y accuracy entre un modelo original y su versión quantizada.

    Args:
        model_fp (nn.Module): Modelo original (float32).
        model_int8 (nn.Module): Modelo quantizado (int8).
        test_loader (DataLoader): DataLoader de test.
        device (str): Dispositivo para evaluación.
        criterion (nn.Module, opcional): Función de pérdida.

    Returns:
        Dict[str, Any]: Diccionario con tamaños (bytes), accuracy y diferencia.
    """
    import tempfile
    import os
    # Guardar temporalmente ambos modelos
    with tempfile.NamedTemporaryFile(delete=False) as f_fp, tempfile.NamedTemporaryFile(delete=False) as f_int8:
        torch.save(model_fp.state_dict(), f_fp.name)
        torch.save(model_int8.state_dict(), f_int8.name)
        size_fp = os.path.getsize(f_fp.name)
        size_int8 = os.path.getsize(f_int8.name)

    # Evaluar accuracy
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
    # Limpiar archivos temporales
    os.remove(f_fp.name)
    os.remove(f_int8.name)
    return {
        'size_fp32_bytes': size_fp,
        'size_int8_bytes': size_int8,
        'accuracy_fp32': acc_fp,
        'accuracy_int8': acc_int8,
        'accuracy_diff': acc_fp - acc_int8
    }
