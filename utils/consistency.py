"""
consistency.py

Numerical equivalence checks across engines (PyTorch, ONNX Runtime, OpenVINO).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compare_logits_torch_ort_ov(model_torch,
                                sess_ort,
                                ov_compiled,
                                sample_np: np.ndarray,
                                tol_fp32: float = 1e-3,
                                tol_fp16: float = 5e-2) -> Dict[str, object]:
    """
    Compare logits among PyTorch, ONNX Runtime and OpenVINO.

    Returns dict with: max_abs_diff_torch_ort, max_abs_diff_torch_ov,
    top1_agree_torch_ort, top1_agree_torch_ov, dtype, consistency_ok.
    """
    model_torch.eval()
    with torch.no_grad():
        torch_out = model_torch(torch.from_numpy(sample_np).to(next(model_torch.parameters()).device))
    torch_np = _to_numpy(torch_out)

    # ORT
    inp_name = sess_ort.get_inputs()[0].name
    ort_out = sess_ort.run(None, {inp_name: sample_np})[0]

    # OpenVINO
    ov_req = ov_compiled.create_infer_request()
    ov_out = ov_req.infer({ov_compiled.inputs[0]: sample_np})[ov_compiled.outputs[0]]

    max_diff_ort = float(np.max(np.abs(torch_np - ort_out)))
    max_diff_ov = float(np.max(np.abs(torch_np - ov_out)))
    top1_t = np.argmax(torch_np, axis=1)
    top1_o = np.argmax(ort_out, axis=1)
    top1_v = np.argmax(ov_out, axis=1)
    t1_agree_ort = bool(np.all(top1_t == top1_o))
    t1_agree_ov = bool(np.all(top1_t == top1_v))

    # Guess dtype from model outputs
    dtype = str(torch_np.dtype)
    tol = tol_fp16 if 'float16' in dtype or 'half' in dtype else tol_fp32
    ok = (max_diff_ort <= tol) and (max_diff_ov <= tol)

    return {
        'max_abs_diff_torch_ort': max_diff_ort,
        'max_abs_diff_torch_ov': max_diff_ov,
        'top1_agree_torch_ort': t1_agree_ort,
        'top1_agree_torch_ov': t1_agree_ov,
        'dtype': dtype,
        'consistency_ok': bool(ok)
    }
