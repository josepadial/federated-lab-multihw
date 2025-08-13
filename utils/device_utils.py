"""
device_utils.py

Utilities for device detection, selection, and naming for deep learning projects.
Supports PyTorch, OpenVINO, and ONNX backends.
"""

import logging
import platform
import subprocess
from typing import List, Dict, Optional, Any

import cpuinfo
import numpy as np
import torch
from openvino import Core

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_windows_cpu_name() -> Optional[str]:
    """
    Attempts to retrieve the CPU name on Windows systems using WMIC or PowerShell.
    Returns:
        str or None: The CPU name if found, else None.
    """

    # Try WMIC
    try:
        output = subprocess.check_output(['wmic', 'cpu', 'get', 'Name'], encoding='utf-8')
        lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            cpu_name = lines[1]
            if 'Family' not in cpu_name and 'GenuineIntel' not in cpu_name:
                return cpu_name
    except Exception:
        pass
    # Try PowerShell
    try:
        output = subprocess.check_output([
            'powershell', '-Command',
            'Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name'
        ], encoding='utf-8')
        cpu_name = output.strip().split('\n')[0].strip()
        if cpu_name:
            return cpu_name
    except Exception:
        pass
    return None


def get_cpu_name() -> str:
    """
    Returns the commercial name of the CPU, using platform-specific methods and cpuinfo as fallback.
    Returns:
        str: The CPU name or a generic fallback.
    """

    try:
        if platform.system() == 'Windows':
            cpu_name = get_windows_cpu_name()
            if cpu_name:
                return cpu_name

        cpu_name = cpuinfo.get_cpu_info().get('brand_raw')
        if not cpu_name:
            cpu_name = cpuinfo.get_cpu_info().get('brand')
        if cpu_name:
            return cpu_name
    except Exception:
        pass
    return platform.processor() or platform.uname().processor or "Generic CPU"


def _devices_pytorch() -> List[Dict[str, Any]]:
    devs: List[Dict[str, Any]] = [{'name': get_cpu_name(), 'type': 'CPU', 'id': None}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devs.append({'name': torch.cuda.get_device_name(i), 'type': 'GPU', 'id': i})
    return devs


def _devices_openvino() -> List[Dict[str, Any]]:
    try:
        core = Core()
        return [{'name': d, 'type': get_device_type_openvino(d), 'id': None} for d in core.available_devices]
    except Exception:
        logger.warning("OpenVINO not available; returning CPU only for OV backend")
        return [{'name': 'CPU', 'type': 'CPU', 'id': None}]


def _devices_onnx() -> List[Dict[str, Any]]:
    # Detect ONNX Runtime providers (DirectML, CUDA, CPU)
    try:
        try:
            import onnxruntime_directml as ort  # type: ignore
        except Exception:
            import onnxruntime as ort  # type: ignore
        providers = set(ort.get_available_providers())
    except Exception:
        return [{'name': 'CPU', 'type': 'CPU', 'id': None}]

    devs: List[Dict[str, Any]] = []
    if 'DmlExecutionProvider' in providers:
        devs.append({'name': 'DirectML', 'type': 'GPU', 'id': None})
    if 'CUDAExecutionProvider' in providers:
        gpu_name = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
        devs.append({'name': gpu_name or 'NVIDIA CUDA', 'type': 'GPU', 'id': 0})
    if 'CPUExecutionProvider' in providers or not providers:
        devs.append({'name': 'CPU', 'type': 'CPU', 'id': None})
    return devs


def get_available_devices(backend: str = 'pytorch') -> List[Dict[str, Any]]:
    """
    Detects and returns a list of available devices for the specified backend.
    """
    if backend == 'pytorch':
        return _devices_pytorch()
    if backend == 'openvino':
        return _devices_openvino()
    if backend == 'onnx':
        return _devices_onnx()
    # Default to CPU only
    return [{'name': get_cpu_name(), 'type': 'CPU', 'id': None}]


def get_device_type_openvino(dev_name: str) -> str:
    """
    Maps OpenVINO device name to a device type string.
    Args:
        dev_name (str): OpenVINO device name.
    Returns:
        str: Device type ('CPU', 'GPU', 'NPU', 'VPU', 'OTHER').
    """
    if 'CPU' in dev_name:
        return 'CPU'
    if 'GPU' in dev_name:
        return 'GPU'
    if 'NPU' in dev_name:
        return 'NPU'
    if 'VPU' in dev_name:
        return 'VPU'
    return 'OTHER'


def print_available_devices(backend: str = 'pytorch') -> None:
    """
    Prints the list of available devices for the given backend.
    Args:
        backend (str): Backend to use.
    """
    devices = get_available_devices(backend)
    logger.info(f"Available devices for backend '{backend}':")
    for d in devices:
        id_str = f" (id {d['id']})" if d['id'] is not None else ""
        logger.info(f"- {d['name']} [{d['type']}{id_str}]")


def select_main_device(devices: List[Dict[str, Any]]) -> Any:
    """
    Selects the main device from a list of devices, prioritizing GPU > NPU > VPU > CPU > other.
    Args:
        devices (List[Dict]): List of device info dicts.
    Returns:
        Device object or name, depending on backend.
    """
    # Priority: GPU > NPU > VPU > CPU > other
    for dev_type in ['GPU', 'NPU', 'VPU', 'CPU']:
        for d in devices:
            if d['type'] == dev_type:
                return get_device_object(d)
    # If none of the above, return the first one
    d = devices[0]
    return get_device_object(d)


def get_device_object(d: Dict[str, Any]) -> Any:
    """
    Returns the appropriate device object for the given device dict.
    Args:
        d (Dict): Device info dict.
    Returns:
        Device object or name.
    """
    if d['type'] == 'GPU':
        try:
            return torch.device(f"cuda:{d['id']}") if d['id'] is not None else torch.device('cuda'), d['name'], d[
                'type'], d['id']
        except ImportError:
            pass
    if d['type'] == 'CPU':
        try:
            return torch.device('cpu'), d['name'], d['type'], d['id']
        except ImportError:
            pass
    # For NPU, VPU or others, just return the name
    return d['name'], d['name'], d['type'], d['id']


def get_eval_devices(devices: List[Dict[str, Any]]) -> List[Any]:
    """
    Returns a list of device objects for evaluation, deduplicated.
    Args:
        devices (List[Dict]): List of device info dicts.
    Returns:
        List: List of device objects or names.
    """
    eval_devices = []
    for d in devices:
        if d['type'] == 'GPU':
            try:
                eval_devices.append(torch.device(f"cuda:{d['id']}") if d['id'] is not None else torch.device('cuda'))
            except ImportError:
                eval_devices.append(d['name'])
        elif d['type'] == 'CPU':
            try:
                eval_devices.append(torch.device('cpu'))
            except ImportError:
                eval_devices.append(d['name'])
        elif d['type'] in ['NPU', 'VPU']:
            eval_devices.append(d['name'])
    return list({str(dev): dev for dev in eval_devices}.values())


def _get_fullname_from_torch_device(dev: Any, devices: List[Dict[str, Any]]) -> Optional[str]:
    """
    Attempts to get the full commercial name of a torch.device from the devices list.
    Args:
        dev: torch.device object.
        devices: List of detected device dicts.
    Returns:
        str or None: The device name if found, else None.
    """
    if not hasattr(dev, 'type'):
        return None

    dev_type = dev.type
    if dev_type == 'cuda':
        dev_index = getattr(dev, 'index', None)
        return _find_gpu_name_by_index(devices, dev_index)
    elif dev_type == 'cpu':
        return _find_cpu_name(devices)
    return None


def _find_gpu_name_by_index(devices: List[Dict[str, Any]], dev_index: Optional[int]) -> Optional[str]:
    """
    Helper to find GPU name by index from devices list.
    """
    for d in devices:
        if not (isinstance(d, dict) and d.get('type') == 'GPU'):
            continue
        if dev_index is not None:
            if d.get('id') == dev_index:
                return d.get('name')
        else:
            return d.get('name')
    return None


def _find_cpu_name(devices: List[Dict[str, Any]]) -> Optional[str]:
    """
    Helper to find CPU name from devices list.
    """
    for d in devices:
        if isinstance(d, dict) and d.get('type') == 'CPU':
            return get_cpu_name()
    return None


def _get_fullname_from_str_device(dev: str, devices: List[Dict[str, Any]]) -> Optional[str]:
    """
    Attempts to get the full commercial name of a device from the devices list by string name.
    Args:
        dev (str): Device name string.
        devices (List[Dict]): List of detected device dicts.
    Returns:
        str or None: The device name if found, else None.
    """
    for d in devices:
        if isinstance(d, dict) and d.get('name') == dev:
            return d.get('name')
    return None


def _get_fallback_fullname(dev: Any) -> str:
    """
    Fallback method to get the device name if not found in the devices list.
    Args:
        dev: Device object or string.
    Returns:
        str: The best-effort device name.
    """
    if hasattr(dev, 'type'):
        if dev.type == 'cuda':
            try:
                return torch.cuda.get_device_name(dev)
            except Exception:
                pass
        elif dev.type == 'cpu':
            return get_cpu_name()
    try:
        return cpuinfo.get_cpu_info().get('brand_raw')
    except Exception:
        return platform.processor() or platform.uname().processor or 'CPU'


def get_device_fullname(dev: Any, devices: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Returns the real commercial name of the device (CPU, GPU, NPU, etc) using the detected devices list.
    Uses get_cpu_name logic for CPUs in Windows.
    Args:
        dev: Device object or string.
        devices: List of detected device dicts (optional).
    Returns:
        str: The device's commercial name.
    """
    if devices is not None:
        if hasattr(dev, 'type'):
            name = _get_fullname_from_torch_device(dev, devices)
            if name:
                return name
        elif isinstance(dev, str):
            name = _get_fullname_from_str_device(dev, devices)
            if name:
                return name
    return _get_fallback_fullname(dev)


def get_dummy_input(batch_size=1, shape=(3, 32, 32), dtype='float32', seed=None):
    """
    Returns a dummy input numpy array for benchmarking model inference.
    Uses numpy.random.Generator for reproducibility and modern API.
    Args:
        batch_size (int): Batch size for input.
        shape (tuple): Shape of a single input (channels, height, width).
        dtype (str): Data type of the array.
        seed (int, optional): Seed for reproducibility.
    Returns:
        np.ndarray: Dummy input array of shape (batch_size, *shape).
    """
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((batch_size, *shape))
    return arr.astype(dtype)


def get_gpu_name_and_driver() -> tuple[str, str]:
    """Return (gpu_name, driver_version) using nvidia-smi when available, else ("N/A","N/A")."""
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'
        ], encoding='utf-8')
        line = out.strip().split('\n')[0]
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return 'N/A', 'N/A'


def get_os_string() -> str:
    try:
        return platform.platform()
    except Exception:
        return platform.system()
