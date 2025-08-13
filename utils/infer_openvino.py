"""
infer_openvino.py

Inference helpers for OpenVINO CPU/GPU/NPU with fair measurement and optional accuracy.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from openvino import Core

from .logging_utils import get_logger


def _compile_model(ir_path: str, device: str):
    logger = get_logger("infer_openvino")
    core = Core()
    logger.info("Reading IR: %s", ir_path)
    model = core.read_model(ir_path)
    logger.info("Compiling model for device: %s", device)
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
    for _ in range(warmup):
        infer.infer({input_port: x})
    # Timed
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
