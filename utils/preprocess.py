"""
preprocess.py

Unified preprocessing to produce NCHW float32 normalized tensors from raw images.
Assumes CIFAR-10 stats by default.
"""

from __future__ import annotations

import numpy as np


def preprocess_np(x_raw: np.ndarray, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)) -> np.ndarray:
    """
    Convert raw images to normalized NCHW float32.

    Accepts NHWC (H,W,C) or NCHW input; if HWC/NHWC, transposes to NCHW. Values
    in [0,255] (uint8) or [0,1] (float). Returns float32 NCHW normalized.
    """
    x = x_raw
    if x.ndim == 3:  # single image -> add batch
        x = x[None, ...]
    # If channels last, move to NCHW
    if x.shape[-1] in (1, 3):
        x = np.transpose(x, (0, 3, 1, 2))
    x = x.astype(np.float32)
    # Scale if in 0..255
    if x.max() > 1.0:
        x = x / 255.0
    mean_arr = np.array(mean, dtype=np.float32)[None, :, None, None]
    std_arr = np.array(std, dtype=np.float32)[None, :, None, None]
    x = (x - mean_arr) / std_arr
    return x
