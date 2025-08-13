federated-lab-multihw
======================

Clean, reproducible benchmarking across PyTorch, ONNX Runtime (CPU/CUDA), and OpenVINO on Windows with a single virtual environment.

Notebook flow
-------------
- 00_check_env_and_tests.ipynb — validate environment, providers, and run smoke tests; writes reports/ artifacts.
- 01_train_and_baseline.ipynb — train on PyTorch (CUDA if available), export ONNX, and record PyTorch CPU/CUDA baselines.
- 02_inference_engines_bench.ipynb — benchmark PyTorch/ORT/OpenVINO fairly with unified preprocessing and NVML energy only on CUDA; caching avoids duplicate runs.
- 03_flower_simulation.ipynb — placeholder; not part of the benchmarking flow yet.
- 04_results_and_plots.ipynb — aggregate CSVs and produce plots/tables.

Windows setup (single venv)
---------------------------
- Install PyTorch and torchvision for your CUDA using the official index URL.
- Install onnxruntime-gpu (provides CPU and CUDA providers) and OpenVINO wheels.
- Optional: nvidia-ml-py3 for NVML-based GPU energy measurement.

Quick start
-----------
1) Run 00 to verify environment and providers; fix any issues before continuing.
2) Run 01 to train chosen models and export ONNX.
3) Run 02 to benchmark across engines; results go to a single CSV with caching.
4) Run 04 to generate plots/tables from the CSV(s).

Notes
-----
- Energy is only measured via NVML on NVIDIA GPUs in 02; CPU/iGPU/NPU are recorded as "N/D".
- OpenVINO devices are detected at runtime; unavailable devices are skipped and noted in the CSV.
- See ADR/0001 and reports/AUDIT_REPORT.md for architecture decisions and audit findings.
