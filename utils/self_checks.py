"""
self_checks.py

Smoke tests and quick consistency checks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .preprocess import preprocess_np


def ensure_tiny_onnx(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    from onnx import helper, TensorProto
    # Model: input (1,3,32,32) -> GlobalAveragePool -> Flatten -> Gemm(3->10)
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    W = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[3, 10], vals=[0.1] * 30)
    B = helper.make_tensor(name="B", data_type=TensorProto.FLOAT, dims=[10], vals=[0.0] * 10)
    gap = helper.make_node("GlobalAveragePool", inputs=["input"], outputs=["gap"])
    flat = helper.make_node("Flatten", inputs=["gap"], outputs=["flat"], axis=1)
    gemm = helper.make_node("Gemm", inputs=["flat", "W", "B"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)
    graph = helper.make_graph([gap, flat, gemm], "tiny32", [inp], [out], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path.write_bytes(model.SerializeToString())
    return path


def run_smoke_from_python(project_root: str | Path = ".") -> Dict[str, object]:
    root = Path(project_root)
    reports = root / "reports";
    reports.mkdir(parents=True, exist_ok=True)
    tiny = ensure_tiny_onnx(root / "tests" / "assets" / "tiny.onnx")
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype("float32")
    x = preprocess_np(np.transpose(x, (0, 2, 3, 1)))
    res = {"ort_cpu": {"ok": False}, "ort_cuda": {"ok": False}, "ov_cpu": {"ok": False}}
    # ORT CPU
    try:
        import onnxruntime as ort
        t0 = time.perf_counter()
        sess = ort.InferenceSession(tiny.as_posix(), providers=["CPUExecutionProvider"])  # type: ignore
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        res["ort_cpu"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
        if "CUDAExecutionProvider" in ort.get_available_providers():
            t0 = time.perf_counter()
            sess = ort.InferenceSession(tiny.as_posix(), providers=["CUDAExecutionProvider"])  # type: ignore
            out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            dt = (time.perf_counter() - t0) * 1000.0
            res["ort_cuda"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
    except Exception as e:
        res["ort_error"] = str(e)
    # OpenVINO CPU
    try:
        from openvino import Core
        core = Core()
        model = core.read_model(tiny.as_posix())
        compiled = core.compile_model(model, 'CPU')
        req = compiled.create_infer_request()
        t0 = time.perf_counter()
        out = req.infer({compiled.inputs[0]: x})[compiled.outputs[0]]
        dt = (time.perf_counter() - t0) * 1000.0
        res["ov_cpu"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
    except Exception as e:
        res["ov_error"] = str(e)
    (reports / "smoke_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def quick_consistency_check(onnx_path: str | Path, torch_pt_path: Optional[str | Path] = None) -> Dict[str, object]:
    from .consistency import compare_logits_torch_ort_ov
    from .preprocess import preprocess_np
    import torch
    import onnxruntime as ort
    from openvino import Core

    onnx_path = Path(onnx_path)
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype("float32")
    x = preprocess_np(np.transpose(x, (0, 2, 3, 1)))

    # Torch model optional
    model = None
    if torch_pt_path and Path(torch_pt_path).exists():
        # Attempt to create a simple CNN from project models
        try:
            if str(Path(__file__).resolve().parents[1]) not in sys.path:
                sys.path.append(str(Path(__file__).resolve().parents[1]))
            from models.cnn import CNN
            model = CNN()
            state = torch.load(str(torch_pt_path), map_location='cpu')
            state = state.get('model_state_dict', state)
            model.load_state_dict(state, strict=False)
        except Exception:
            model = None

    # ORT session
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
    # OV compiled
    core = Core();
    ov_model = core.read_model(onnx_path.as_posix());
    ov_comp = core.compile_model(ov_model, 'CPU')

    if model is None:
        # Compare ORT vs OV using ORT as reference (subset of fields)
        out_ort = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        req = ov_comp.create_infer_request();
        out_ov = req.infer({ov_comp.inputs[0]: x})[ov_comp.outputs[0]]
        top1_o = int(np.argmax(out_ort, axis=1)[0]);
        top1_v = int(np.argmax(out_ov, axis=1)[0])
        res = {
            'max_abs_diff_torch_ort': '', 'max_abs_diff_torch_ov': float(np.max(np.abs(out_ort - out_ov))),
            'top1_agree_torch_ort': '', 'top1_agree_torch_ov': bool(top1_o == top1_v),
            'dtype': str(out_ort.dtype), 'consistency_ok': bool(np.allclose(out_ort, out_ov, atol=1e-3))
        }
    else:
        model.eval()
        res = compare_logits_torch_ort_ov(model, sess, ov_comp, x)
    return res
