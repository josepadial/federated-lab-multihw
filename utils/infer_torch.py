"""
infer_torch.py

Inference helpers for PyTorch CPU/CUDA: latency and throughput with warmup, optional accuracy.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
def _warmup(model: torch.nn.Module, data: DataLoader, device: torch.device, steps: int) -> None:
    it = iter(data)
    for _ in range(min(steps, len(data))):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        _ = _run_model(model, x, device)
        if device.type == "cuda":
            torch.cuda.synchronize()


def _accumulate(preds: np.ndarray, y: np.ndarray) -> int:
    return int((preds == y).sum())



@torch.no_grad()
def _run_model(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    model.to(device)
    return model(x.to(device))


def benchmark_dataloader(
        model: torch.nn.Module,
        data: DataLoader,
        device: torch.device,
        warmup: int = 10,
        runs: int = 100,
        measure_accuracy: bool = True,
) -> Dict[str, float]:
    """
    Measure average latency and throughput (items/s) over a dataloader.
    """
    latencies: list[float] = []
    preds_correct = 0
    total = 0

    # Warmup
    _warmup(model, data, device, warmup)

    # Timed runs
    for i, (x, y) in enumerate(data):
        if i >= runs:
            break
        t0 = time.perf_counter()
        out = _run_model(model, x, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        latencies.append(dt)
        total += y.shape[0]
        if measure_accuracy:
            pred = out.argmax(1).detach().cpu().numpy()
            preds_correct += _accumulate(pred, y.numpy())

    lat_ms = np.array(latencies) * 1000.0
    lat_ms_mean = float(lat_ms.mean()) if lat_ms.size else 0.0
    lat_ms_p95 = float(np.percentile(lat_ms, 95)) if lat_ms.size else 0.0
    thr = float(total / sum(latencies)) if latencies else 0.0
    acc = float(preds_correct / total) if measure_accuracy and total else 0.0
    return {"lat_ms_mean": lat_ms_mean, "lat_ms_p95": lat_ms_p95, "thr_ips": thr, "acc": acc}
