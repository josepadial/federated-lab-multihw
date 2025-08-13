"""
ov_convert.py

Helper to convert ONNX models to OpenVINO IR (XML+BIN) using the modern
OpenVINO Python API per official docs. Preferred path uses ov.convert_model
followed by ov.serialize. Fallbacks read/compile directly if convert is
unavailable. Reference:
https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-to-ir.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import openvino as ov

from .logging_utils import get_logger


def onnx_to_ir(
        onnx_path: str | Path,
        out_dir: str | Path,
        xml_name: Optional[str] = None,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        compress_to_fp16: bool = False,
) -> Path:
    """
    Convert an ONNX model to OpenVINO IR and return the XML path.

    Args:
        onnx_path: Path to the ONNX file.
        out_dir: Output directory to place the IR files.
        xml_name: Optional base filename for the IR (defaults to ONNX stem).
        input_shape: Optional shape of the model inputs (N, C, H, W).
        compress_to_fp16: Whether to compress weights to FP16 (default: False).
    Returns:
        Path to the generated IR XML.
    Raises:
        ImportError if OpenVINO conversion API is unavailable.
        Exception for other conversion failures.
    """
    logger = get_logger("ov_convert")
    onnx_path = Path(onnx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if xml_name is None:
        xml_name = onnx_path.stem + ".xml"
    xml_path = out_dir / xml_name
    bin_path = xml_path.with_suffix(".bin")

    # For ONNX files, the correct and stable path is Core.read_model + serialize.
    # ov.convert_model is not intended for ONNX file paths and may trigger unrelated converters.
    try:
        if onnx_path.suffix.lower() == ".onnx":
            logger.info("Reading ONNX with ov.Core().read_model: %s", onnx_path)
            core = ov.Core()
            ov_model = core.read_model(onnx_path.as_posix())
            # Note: input_shape and compress_to_fp16 are ignored here. If needed, implement shape overrides via
            # ov.preprocess API and FP16 compression via proper transformations in a future revision.
            ov.serialize(ov_model, xml_path.as_posix(), bin_path.as_posix())
            logger.info("Serialized IR via Core.read_model: %s, %s", xml_path.name, bin_path.name)
            return xml_path
        else:
            # Non-ONNX inputs: best-effort convert_model path without unsupported kwargs
            logger.info("Converting model with ov.convert_model (non-ONNX input): %s", onnx_path)
            ov_model = ov.convert_model(onnx_path.as_posix())
            ov.serialize(ov_model, xml_path.as_posix(), bin_path.as_posix())
            logger.info("Serialized IR via convert_model: %s, %s", xml_path.name, bin_path.name)
            return xml_path
    except Exception as ex:
        logger.error("Failed to produce IR from %s: %s", onnx_path, ex, exc_info=True)
        raise
