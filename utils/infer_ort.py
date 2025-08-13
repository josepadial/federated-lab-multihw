"""
infer_ort.py

Inference helpers for ONNX Runtime CPU/CUDA with fair measurement and optional accuracy.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from .logging_utils import get_logger

try:
    import onnxruntime as ort
except Exception:  # Allow DirectML installations to be used as onnxruntime
    import onnxruntime_directml as ort  # type: ignore


def _make_session(onnx_path: str, provider: str) -> ort.InferenceSession:
    logger = get_logger("infer_ort")
    providers = [provider]
    logger.info("Creating ORT session: %s | provider=%s", onnx_path, provider)
    return ort.InferenceSession(onnx_path, providers=providers)


def benchmark_numpy(
        onnx_path: str,
        x: np.ndarray,
        provider: str = "CPUExecutionProvider",
        warmup: int = 10,
        runs: int = 100,
        y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    logger = get_logger("infer_ort")
    sess = _make_session(onnx_path, provider)
    inp_name = sess.get_inputs()[0].name
    lat = []
    correct = 0
    total = 0
    # Warmup
    for _ in range(warmup):
        sess.run(None, {inp_name: x})
    # Timed
    for _ in range(runs):
        t0 = time.perf_counter()
        out = sess.run(None, {inp_name: x})[0]
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
    logger.info("ORT metrics: %s", metrics)
    return metrics
