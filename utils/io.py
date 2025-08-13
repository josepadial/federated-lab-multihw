"""
io.py

Lightweight I/O helpers: CSV append-with-header, model hash, and system info.
"""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

CSV_SCHEMA = [
    "ts", "exp_id", "model", "dataset", "precision", "engine", "provider",
    "batch", "warmup", "runs", "lat_ms_mean", "lat_ms_p95", "thr_ips", "acc",
    "energy_j", "cached", "device_name", "cpu_name", "gpu_name", "os",
    "torch_ver", "ort_ver", "ov_ver", "driver_ver", "model_hash",
    "consistency_ok", "max_abs_diff_torch_ort", "max_abs_diff_torch_ov",
    "top1_agree_torch_ort", "top1_agree_torch_ov"
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def csv_append_row(csv_path: str, row: Dict[str, object], schema: Optional[List[str]] = None) -> None:
    schema = schema or CSV_SCHEMA
    ensure_dir(os.path.dirname(csv_path) or ".")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        if not file_exists:
            writer.writeheader()
        # Only keep expected keys; missing keys default to ''
        filtered = {k: row.get(k, "") for k in schema}
        writer.writerow(filtered)


def csv_has_row_with(csv_path: str, match: Dict[str, object]) -> bool:
    if not os.path.isfile(csv_path):
        return False
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    mask = None
    for k, v in match.items():
        if k not in df.columns:
            return False
        col_mask = df[k] == v
        mask = col_mask if mask is None else (mask & col_mask)
    return bool(mask is not None and mask.any())


def git_commit_short() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "N/A"


def nvidia_driver_version() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip().split("\n")[0]
    except Exception:
        return "N/A"


def runtime_versions():
    """Return versions of torch, onnxruntime, openvino, and OS string."""
    try:
        import torch  # type: ignore
        tv = torch.__version__
    except Exception:
        tv = "N/A"
    try:
        import onnxruntime as ort  # type: ignore
        ortv = ort.__version__
    except Exception:
        ortv = "N/A"
    try:
        import openvino as _ov  # type: ignore
        ovv = _ov.__version__
    except Exception:
        ovv = "N/A"
    import platform as _pf
    return {"torch_ver": tv, "ort_ver": ortv, "ov_ver": ovv, "os": _pf.platform()}


def build_cache_key(model_hash: str, engine: str, provider: str, precision: str,
                    batch: int, warmup: int, runs: int,
                    torch_ver: str, ort_ver: str, ov_ver: str, driver_ver: str) -> str:
    payload = f"{model_hash}|{engine}|{provider}|{precision}|{batch}|{warmup}|{runs}|{torch_ver}|{ort_ver}|{ov_ver}|{driver_ver}"
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
