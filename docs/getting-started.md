# Getting Started

This guide helps you set up a single Python environment on Windows and run the notebooks in order.

Prerequisites
- Windows 10/11 with an up-to-date NVIDIA driver (for CUDA energy measurement)
- Python 3.11 (recommended)
- Git, PowerShell

Environment setup (single venv)
1) Create and activate a virtual environment.
2) Install PyTorch and torchvision appropriate for your CUDA or CPU-only.
3) Install ONNX Runtime (GPU build provides CPU and CUDA).
4) Install OpenVINO runtime.
5) Optional: nvidia-ml-py3 for NVML energy measurement.

Notebook workflow
- 00: Validate providers, generate reports/providers_status.csv and a tiny consistency check.
- 01: Train selected models (CNN, MLP, MobileNetV3, EfficientNet-Lite0) and export ONNX.
- 02: Benchmark inference across engines/providers; metrics go to metrics/infer_metrics.csv.
- 04: Produce plots and tables from the CSVs under reports/.

Tips
- Use LOG_LEVEL=INFO for concise logs; DEBUG for detailed traces.
- If a provider (e.g., CUDA, OpenVINO GPU) is missing, the notebook records an informative row instead of failing.

Next: [Reproducibility](./reproducibility.md)
