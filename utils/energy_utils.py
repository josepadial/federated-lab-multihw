# energy_utils.py

import time

import numpy as np


def measure_energy_nvidia_gpu(infer_fn, runs=100):
    """
    Measures energy consumption for NVIDIA GPU inference using nvidia-smi.
    Queries instantaneous power draw in Watts before each run.
    Args:
        infer_fn: Callable performing one inference.
        runs (int): Number of inference runs.
    Returns:
        float: Estimated total energy in Joules.
    """
    import subprocess
    energy_samples = []
    for _ in range(runs):
        # Query current GPU power draw
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL
            )
            power = float(out.decode().strip())
        except Exception:
            power = 30.0  # fallback estimate in Watts
        t0 = time.perf_counter()
        infer_fn()
        t1 = time.perf_counter()
        energy_samples.append((t1 - t0) * power)
    return float(np.sum(energy_samples))
