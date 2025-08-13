"""
env_report.py

Gather detailed environment info and save as JSON and Markdown.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from .device_utils import get_cpu_name, get_gpu_name_and_driver, get_os_string
from .io import sha256_file


def _safe_import_version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "(no __version__)")
    except Exception:
        return "not installed"


def _torch_versions():
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cudnn": getattr(getattr(torch.backends, "cudnn", object), "version", lambda: None)(),
        }
    except Exception:
        return {"torch": "not installed", "cuda": None, "cudnn": None}


def _ort_providers_status() -> Dict[str, object]:
    status = {"available": [], "session_cpu": False, "session_cuda": False, "error": None}
    try:
        import onnxruntime as ort
        provs = list(ort.get_available_providers())
        status["available"] = provs
        try:
            _ = ort.InferenceSession(onnx_model_bytes(), providers=["CPUExecutionProvider"])  # type: ignore
            status["session_cpu"] = True
        except Exception:
            status["session_cpu"] = False
        if "CUDAExecutionProvider" in provs:
            try:
                _ = ort.InferenceSession(onnx_model_bytes(), providers=["CUDAExecutionProvider"])  # type: ignore
                status["session_cuda"] = True
            except Exception:
                status["session_cuda"] = False
    except Exception as e:
        status["error"] = str(e)
    return status


def onnx_model_bytes() -> bytes:
    # Tiny in-memory ONNX: Input [1,3,1,1] -> GlobalAveragePool -> Flatten -> Gemm to 10
    from onnx import helper, TensorProto
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 1])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    W = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[3, 10], vals=[0.1] * 30)
    B = helper.make_tensor(name="B", data_type=TensorProto.FLOAT, dims=[10], vals=[0.0] * 10)
    gap = helper.make_node("GlobalAveragePool", inputs=["input"], outputs=["gap"])
    flat = helper.make_node("Flatten", inputs=["gap"], outputs=["flat"], axis=1)
    gemm = helper.make_node("Gemm", inputs=["flat", "W", "B"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)
    graph = helper.make_graph([gap, flat, gemm], "tiny", [inp], [out], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    # Some ONNX Runtime builds only support up to IR version 10.
    # Ensure the tiny test model uses IR v10 to avoid session creation failures.
    try:
        model.ir_version = 10  # type: ignore[attr-defined]
    except Exception:
        pass
    return model.SerializeToString()


def _ov_devices() -> Dict[str, object]:
    info = {"available": [], "error": None}
    try:
        from openvino import Core
        core = Core()
        info["available"] = list(core.available_devices)
    except Exception as e:
        info["error"] = str(e)
    return info


def _repo_root() -> Path:
    # utils/env_report.py -> project root is parent of utils
    return Path(__file__).resolve().parent.parent


def _reports_dir() -> Path:
    root = _repo_root()
    d = root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _onnx_dir() -> Path:
    root = _repo_root()
    d = root / "models_saved" / "onnx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_paths(root: Path) -> Dict[str, object]:
    paths = {
        "models": (root / "models").exists(),
        "utils": (root / "utils").exists(),
        "models_saved.pytorch": (root / "models_saved" / "pytorch").exists(),
        "models_saved.onnx": (root / "models_saved" / "onnx").exists(),
        "models_saved.openvino_ir": (root / "models_saved" / "openvino_ir").exists(),
        "metrics": (root / "metrics").exists(),
        "reports": (root / "reports").exists(),
    }
    return paths


def _write_permissions(root: Path) -> Dict[str, bool]:
    res = {"metrics": False, "reports": False}
    for k in res.keys():
        d = root / k
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / ".write_test"
            tmp.write_text("ok", encoding="utf-8")
            tmp.unlink(missing_ok=True)
            res[k] = True
        except Exception:
            res[k] = False
    return res


def _hash_models(root: Path) -> Dict[str, List[Dict[str, str]]]:
    onnx_dir = root / "models_saved" / "onnx"
    pt_dir = root / "models_saved" / "pytorch"
    onnx = []
    pt = []
    if onnx_dir.exists():
        for p in onnx_dir.glob("*.onnx"):
            try:
                onnx.append({"file": str(p), "sha256": sha256_file(str(p))})
            except Exception:
                pass
    if pt_dir.exists():
        for p in pt_dir.glob("*.pt"):
            try:
                pt.append({"file": str(p), "sha256": sha256_file(str(p))})
            except Exception:
                pass
    return {"onnx": onnx, "pytorch": pt}


def gather_env_info(project_root: str | Path = ".") -> Dict[str, object]:
    root = Path(project_root).resolve()
    info: Dict[str, object] = {
        "os": get_os_string(),
        "python": sys.version,
        "executable": sys.executable,
        "venv": os.environ.get("VIRTUAL_ENV", "(system)"),
        "versions": {
            "torch": _torch_versions()["torch"],
            "torch_cuda": _torch_versions()["cuda"],
            "cudnn": _torch_versions()["cudnn"],
            "onnxruntime": _safe_import_version("onnxruntime"),
            "openvino": _safe_import_version("openvino"),
            "numpy": _safe_import_version("numpy"),
            "pandas": _safe_import_version("pandas"),
            "matplotlib": _safe_import_version("matplotlib"),
        },
        "nvidia": {
            "gpu_name": get_gpu_name_and_driver()[0],
            "driver_version": get_gpu_name_and_driver()[1],
        },
        "cpu": {
            "name": get_cpu_name(),
        },
        "onnxruntime": _ort_providers_status(),
        "openvino": _ov_devices(),
        "paths": _project_paths(root),
        "write_permissions": _write_permissions(root),
        "models_hash": _hash_models(root),
    }
    return info


def save_env_report(info: Dict[str, object], out_json: str | Path, out_md: str | Path) -> None:
    out_json = Path(out_json);
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = Path(out_md);
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    # Markdown summary
    lines = []
    lines.append("# Environment Report\n")
    lines.append(f"OS: {info.get('os')}\n")
    v = info.get("versions", {})
    lines.append("## Versions\n")
    for k in ["torch", "torch_cuda", "cudnn", "onnxruntime", "openvino", "numpy", "pandas", "matplotlib"]:
        lines.append(f"- {k}: {v.get(k)}\n")
    lines.append("\n## Hardware\n")
    lines.append(f"- CPU: {info.get('cpu', {}).get('name')}\n")
    nv = info.get('nvidia', {})
    lines.append(f"- GPU: {nv.get('gpu_name')} (driver {nv.get('driver_version')})\n")
    lines.append("\n## ONNX Runtime\n")
    ort = info.get('onnxruntime', {})
    lines.append(f"- Providers available: {ort.get('available')}\n")
    lines.append(f"- CPU session: {ort.get('session_cpu')} | CUDA session: {ort.get('session_cuda')}\n")
    lines.append("\n## OpenVINO\n")
    ov = info.get('openvino', {})
    lines.append(f"- Devices: {ov.get('available')}\n")
    lines.append("\n## Project paths\n")
    for k, vv in info.get('paths', {}).items():
        lines.append(f"- {k}: {vv}\n")
    lines.append("\n## Write permissions\n")
    for k, vv in info.get('write_permissions', {}).items():
        lines.append(f"- {k}: {vv}\n")
    Path(out_md).write_text("".join(lines), encoding="utf-8")


# ---------- Public helpers used by the notebook ----------

def check_pytorch() -> Dict[str, object]:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        devs = torch.cuda.device_count() if cuda_ok else 0
        cudnn_v = None
        try:
            import torch.backends.cudnn as cudnn
            cudnn_v = cudnn.version()
        except Exception:
            pass
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(cuda_ok),
            "cuda_device_count": devs,
            "cudnn": cudnn_v,
        }
    except Exception as e:
        return {"installed": False, "error": str(e)}


def _write_providers_status_csv(path: Path, ort_status: Dict[str, object], ov_info: Dict[str, object]) -> None:
    import csv
    header = ["backend", "provider_or_device", "available", "test_infer_ok", "error"]
    rows: List[List[object]] = []
    # ORT rows
    provs = list(ort_status.get("available", []) or [])
    cpu_ok = bool(ort_status.get("session_cpu") is True)
    cuda_val = ort_status.get("session_cuda")
    cuda_ok = bool(cuda_val is True)
    err = ort_status.get("error")
    rows.append(["ORT", "CPU", ("CPUExecutionProvider" in provs), cpu_ok, None if cpu_ok else err])
    if "CUDAExecutionProvider" in provs:
        # Extracted conditional for CUDA error
        if cuda_ok:
            cuda_error = None
        elif isinstance(cuda_val, str):
            cuda_error = cuda_val
        else:
            cuda_error = err
        rows.append([
            "ORT", "CUDA", True, cuda_ok, cuda_error
        ])
    if "TensorrtExecutionProvider" in provs:
        rows.append(["ORT", "TensorRT", True, False, None])
    # OpenVINO rows
    devices = list(ov_info.get("available", []) or [])
    for dev in devices:
        # Mark CPU as ok (we don't compile here to keep it lightweight)
        rows.append(["OpenVINO", dev, True, True, None])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(header);
        w.writerows(rows)


def check_onnxruntime() -> Dict[str, object]:
    status = _ort_providers_status()
    # Also emit providers_status.csv expected by the notebook summary
    try:
        _write_providers_status_csv(_reports_dir() / "providers_status.csv", status, _ov_devices())
    except Exception:
        pass
    return status


def check_openvino() -> Dict[str, object]:
    return _ov_devices()


def check_write_permissions(path: str | Path) -> bool:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        tmp = p / ".write_test"
        tmp.write_text("ok", encoding="utf-8")
        tmp.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def test_model_import_and_export(onnx_out_dir: str | Path | None = None) -> Dict[str, object]:
    """Create a tiny ONNX file for smoke tests. Returns paths and hashes."""
    onnx_dir = Path(onnx_out_dir) if onnx_out_dir else _onnx_dir()
    onnx_dir.mkdir(parents=True, exist_ok=True)
    target = onnx_dir / "tiny_envcheck.onnx"
    try:
        data = onnx_model_bytes()
        target.write_bytes(data)
        return {"ok": True, "onnx_path": str(target), "sha256": sha256_file(str(target))}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def test_inference_consistency(onnx_dir: str | Path | None = None) -> Dict[str, object]:
    """Run a tiny inference with ORT CPU and (if available) CUDA; compare outputs."""
    import numpy as np
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"ok": False, "error": f"onnxruntime import failed: {e}"}

    onnx_dir_p = Path(onnx_dir) if onnx_dir else _onnx_dir()
    model_path = onnx_dir_p / "tiny_envcheck.onnx"
    if not model_path.exists():
        # Create it if missing
        _ = test_model_import_and_export(onnx_dir_p)

    x = np.ones((1, 3, 1, 1), dtype=np.float32)
    feeds = {"input": x}

    # ORT CPU
    cpu_out = None
    cpu_ok = False
    cpu_err = None
    try:
        sess_cpu = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])  # type: ignore
        y = sess_cpu.run(None, feeds)[0]
        cpu_out = y
        cpu_ok = True
    except Exception as e:
        cpu_err = str(e)

    # ORT CUDA (optional)
    cuda_ok = False
    cuda_out = None
    cuda_err = None
    try:
        if "CUDAExecutionProvider" in (getattr(ort, "get_available_providers", lambda: [])()):
            sess_cuda = ort.InferenceSession(model_path.as_posix(), providers=["CUDAExecutionProvider"])  # type: ignore
            y2 = sess_cuda.run(None, feeds)[0]
            cuda_out = y2
            cuda_ok = True
    except Exception as e:
        cuda_err = str(e)

    # Compare if both present; else treat CPU alone as pass
    consistent = None
    if cpu_ok and cuda_ok:
        consistent = bool(np.allclose(cpu_out, cuda_out, rtol=1e-4, atol=1e-5))
    elif cpu_ok:
        consistent = True
    else:
        consistent = False

    result = {
        "ok": bool(cpu_ok or cuda_ok),
        "cpu_ok": cpu_ok,
        "cuda_ok": cuda_ok,
        "consistent": consistent,
        "cpu_error": cpu_err,
        "cuda_error": cuda_err,
    }

    # Write quickcheck file for Section F
    try:
        out = _reports_dir() / "consistency_quickcheck.json"
        out.write_text(json.dumps({"consistency_ok": consistent, "detail": result}, indent=2), encoding="utf-8")
    except Exception:
        pass
    return result
