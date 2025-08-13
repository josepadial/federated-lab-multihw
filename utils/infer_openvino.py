"""
infer_openvino.py

Inference helpers for OpenVINO CPU/GPU/NPU with fair measurement and optional accuracy.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from openvino import Core
try:
    # Prefer official properties API when available (OV >= 2023.2)
    from openvino import properties as ov_properties  # type: ignore
    from openvino import Type as OvType  # type: ignore
except Exception:  # Fallback for older packages; we'll use string keys
    ov_properties = None
    OvType = None  # type: ignore

from utils.logging_utils import get_logger


def _compile_model(ir_path: str, device: str):
    logger = get_logger("infer_openvino")
    core = Core()
    logger.info("Reading IR: %s", ir_path)
    model = core.read_model(ir_path)
    logger.info("Compiling model for device: %s", device)
    # On some OpenVINO NPU backends, setting an explicit FP32 inference precision hint is
    # necessary to avoid unintended quantization or degraded accuracy.
    if device.upper() == "NPU":
        try:
            if ov_properties is not None and OvType is not None:
                return core.compile_model(
                    model,
                    device,
                    {ov_properties.hint.inference_precision: OvType.f32},
                )
            # Fallback using legacy string key
            return core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f32"})
        except Exception:
            logger.warning("NPU compile with precision hint failed; falling back to default.")
            return core.compile_model(model, device)
    return core.compile_model(model, device)


def benchmark_numpy(
        ir_path: str,
        x: np.ndarray,
        device: str = "CPU",
        warmup: int = 10,
        runs: int = 100,
        y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    logger = get_logger("infer_openvino")
    compiled = _compile_model(ir_path, device)
    infer = compiled.create_infer_request()
    input_port = compiled.inputs[0]
    lat = []
    correct = 0
    total = 0
    # Warmup
    if device.upper() == "NPU" and x.shape[0] != 1:
        # Warmup a few single samples
        for i in range(min(warmup, x.shape[0])):
            _ = infer.infer({input_port: x[i:i+1]})
    else:
        for _ in range(warmup):
            _ = infer.infer({input_port: x})
    # Timed
    if device.upper() == "NPU" and x.shape[0] != 1:
        # Per-sample loop to respect NPU batch=1 while keeping totals comparable
        for _ in range(runs):
            for i in range(x.shape[0]):
                t0 = time.perf_counter()
                out = infer.infer({input_port: x[i:i+1]})[compiled.outputs[0]]
                dt = time.perf_counter() - t0
                lat.append(dt)
                if y_true is not None:
                    pred = out.argmax(1)
                    correct += int((pred == y_true[i:i+1]).sum())
                total += 1
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            out = infer.infer({input_port: x})[compiled.outputs[0]]
            dt = time.perf_counter() - t0
            lat.append(dt)
            if y_true is not None:
                pred = out.argmax(1)
                correct += int((pred == y_true).sum())
                total += y_true.shape[0]
            else:
                total += x.shape[0]
    lat_ms = np.array(lat) * 1000.0
    lat_ms_mean = float(lat_ms.mean()) if lat_ms.size else 0.0
    lat_ms_p95 = float(np.percentile(lat_ms, 95)) if lat_ms.size else 0.0
    thr = float(total / sum(lat)) if lat else 0.0
    acc = float(correct / total) if y_true is not None and total else 0.0
    metrics = {"lat_ms_mean": lat_ms_mean, "lat_ms_p95": lat_ms_p95, "thr_ips": thr, "acc": acc}
    logger.info("OV metrics: %s", metrics)
    return metrics
