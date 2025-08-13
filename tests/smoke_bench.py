"""
Quick smoke test: ensure tiny ONNX and run ORT CPU/CUDA and OpenVINO CPU one pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def ensure_tiny_onnx(path: Path) -> Path:
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


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / 'reports';
    reports.mkdir(parents=True, exist_ok=True)
    tiny = ensure_tiny_onnx(root / 'tests' / 'assets' / 'tiny.onnx')
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype('float32')

    results = {"ort_cpu": {"ok": False}, "ort_cuda": {"ok": False}, "ov_cpu": {"ok": False}}
    rc = 0
    # ORT CPU
    try:
        import onnxruntime as ort
        t0 = time.perf_counter()
        sess = ort.InferenceSession(tiny.as_posix(), providers=['CPUExecutionProvider'])
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        results["ort_cpu"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
        # ORT CUDA if available
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            t0 = time.perf_counter()
            sess = ort.InferenceSession(tiny.as_posix(), providers=['CUDAExecutionProvider'])
            out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            dt = (time.perf_counter() - t0) * 1000.0
            results["ort_cuda"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
    except Exception as ex:
        results["ort_error"] = str(ex)
        rc = max(rc, 1)

    # OpenVINO CPU
    try:
        from openvino import Core
        core = Core();
        model = core.read_model(tiny.as_posix());
        compiled = core.compile_model(model, 'CPU')
        req = compiled.create_infer_request()
        t0 = time.perf_counter()
        out = req.infer({compiled.inputs[0]: x})[compiled.outputs[0]]
        dt = (time.perf_counter() - t0) * 1000.0
        results["ov_cpu"] = {"ok": bool(out.shape[0] == 1), "lat_ms": dt}
    except Exception as ex:
        results["ov_error"] = str(ex)
        rc = max(rc, 1)

    (reports / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    return rc


if __name__ == '__main__':
    sys.exit(main())
