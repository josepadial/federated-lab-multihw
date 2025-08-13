"""
energy.py

Energy measurement helpers. NVIDIA GPU via NVML when available; CPU/NPU return N/D.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Tuple


def measure_energy_nvml(infer_fn: Callable[[], None], runs: int = 100) -> float:
    """
    Measure energy via NVML. If NVML not available, returns -1.0.
    """
    try:
        import pynvml  # type: ignore
    except Exception:
        return -1.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        energy_j = 0.0
        for _ in range(runs):
            p0 = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            t0 = time.perf_counter()
            infer_fn()
            t1 = time.perf_counter()
            p1 = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            power = (p0 + p1) / 2.0
            energy_j += (t1 - t0) * power
        return float(energy_j)
    except Exception:
        return -1.0
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def energy_placeholder() -> str:
    return "N/D"


def _nvml_available() -> bool:
    try:
        import pynvml  # type: ignore
        return True
    except Exception:
        return False


@dataclass
class GpuEnergyMeterNVML:
    device_index: int = 0

    def __post_init__(self):
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            # NVML unavailable; mark as disabled
            self._nvml = None  # type: ignore[attr-defined]
            self._handle = None  # type: ignore[attr-defined]

    def __del__(self):  # best-effort finalize
        try:
            if getattr(self, "_nvml", None) is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass

    def _read_energy_j(self) -> float:
        # NVML exposes total energy in millijoules on supported GPUs (Turing+), else approximate via power draw.
        try:
            if getattr(self, "_nvml", None) is None or getattr(self, "_handle", None) is None:
                return -1.0
            e_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)  # type: ignore
            return float(e_mj) / 1000.0
        except Exception:
            # Fallback: instantaneous power * time integration is not available here; return -1 to signal N/D
            return -1.0

    def measure(self, func: Callable[[], None]) -> Tuple[float, float]:
        import time, torch
        # Sync before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e0 = self._read_energy_j()
        t0 = time.perf_counter()
        func()  # user function should include its own syncs if needed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        e1 = self._read_energy_j()
        if e0 >= 0 and e1 >= 0:
            return max(e1 - e0, 0.0), dt_ms
        return -1.0, dt_ms
