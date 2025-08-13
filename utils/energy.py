"""
energy.py

Energy measurement helpers. NVIDIA GPU via NVML when available; CPU/NPU return N/D.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict
from utils.logging_utils import get_logger


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


# Last measurement snapshot for convenience-based CSV autofill
LAST_ENERGY_STATS: Optional[Dict[str, float]] = None


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
        self._logger = get_logger("energy")
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
        import time, torch, threading
        global LAST_ENERGY_STATS
        # Helper: background power sampler (fallback)
        samples_lock = threading.Lock()
        samples: list[tuple[float, float]] = []  # (timestamp, watts)
        stop_evt = threading.Event()

        def _sampler():
            try:
                while not stop_evt.is_set():
                    t = time.perf_counter()
                    try:
                        p_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # type: ignore[attr-defined]
                    except Exception:
                        break
                    with samples_lock:
                        samples.append((t, p_w))
                    # ~200 Hz sampling
                    time.sleep(0.005)
            except Exception:
                pass

        # Sync before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e0 = self._read_energy_j()
        t0 = time.perf_counter()

        # Start sampler thread (won't hurt if we end up using total energy)
        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()

        func()  # user function should include its own syncs if needed

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        stop_evt.set()
        sampler_thread.join(timeout=0.2)

        e1 = self._read_energy_j()
        if e0 >= 0 and e1 >= 0 and e1 > e0:
            energy_j = max(e1 - e0, 0.0)
            avg_power = (energy_j / (dt_ms / 1000.0)) if dt_ms > 0 else 0.0
            # Heuristic: for very short windows or implausibly high power, prefer sampled integration
            prefer_sampling = (dt_ms < 80.0) or (avg_power > 250.0)
            if not prefer_sampling:
                with contextlib.suppress(Exception):
                    self._logger.info(
                        "gpu-energy: method=nvml_total_energy energy_j=%.6f dt_ms=%.3f avg_power_w=%.3f",
                        energy_j,
                        dt_ms,
                        avg_power,
                    )
                # Save snapshot for downstream CSV autofill
                LAST_ENERGY_STATS = {"energy_j": float(energy_j), "dt_ms": float(dt_ms), "avg_power_w": float(avg_power)}
                return energy_j, dt_ms

        # Fallback: integrate sampled power over time
        with samples_lock:
            s = list(samples)
        if len(s) >= 2:
            energy_j = 0.0
            for i in range(len(s) - 1):
                t_a, p_a = s[i]
                t_b, p_b = s[i + 1]
                dt = max(t_b - t_a, 0.0)
                energy_j += ((p_a + p_b) * 0.5) * dt
            energy_j = float(max(energy_j, 0.0))
            with contextlib.suppress(Exception):
                self._logger.info("gpu-energy: method=power_sampling samples=%d energy_j=%.6f dt_ms=%.3f avg_power_w=%.3f", len(s), energy_j, dt_ms, (energy_j / (dt_ms / 1000.0) if dt_ms > 0 else 0.0))
            # Save snapshot for downstream CSV autofill
            LAST_ENERGY_STATS = {"energy_j": float(energy_j), "dt_ms": float(dt_ms), "avg_power_w": float(energy_j / (dt_ms / 1000.0) if dt_ms > 0 else 0.0)}
            return energy_j, dt_ms

        return -1.0, dt_ms
