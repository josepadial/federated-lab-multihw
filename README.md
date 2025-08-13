federated-lab-multihw
======================

Clean, reproducible benchmarking across PyTorch, ONNX Runtime (CPU/CUDA), and OpenVINO on Windows with a single virtual environment.

Documentation
-------------
- See [README](docs/README.md) for the full documentation set: overview, getting started, reproducibility, benchmarks, results, energy, troubleshooting.

Notebook flow
-------------
- [00_check_env_and_tests.ipynb](notebooks/00_check_env_and_tests.ipynb) — validate environment, providers, and run smoke tests; writes reports/ artifacts.
- [01_train_and_baseline.ipynb](notebooks/01_train_and_baseline.ipynb) — train on PyTorch (CUDA if available), export ONNX, and record PyTorch CPU/CUDA baselines.
- [02_inference_engines_bench.ipynb](notebooks/02_inference_engines_bench.ipynb) — benchmark PyTorch/ORT/OpenVINO fairly with unified preprocessing and NVML energy only on CUDA; caching avoids duplicate runs.
- [03_flower_simulation.ipynb](notebooks/03_flower_simulation.ipynb) — placeholder; not part of the benchmarking flow yet.
- [04_results_and_plots.ipynb](notebooks/04_results_and_plots.ipynb) — aggregate CSVs and produce plots/tables.

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

Interpretation and reflection (Notebooks 01, 02, 04)
----------------------------------------------------
This section summarizes what the current metrics show. Sources: `metrics/train_baseline.csv`, `metrics/engines_bench.csv`, `metrics/*_train_history.csv`, and `logs/run.log`.

- Training dynamics (01): All models converge cleanly on CIFAR-10. Final top-1 on the held-out eval split: EfficientNet-Lite0 ≈86–87%, MobileNetV3 ≈80–81%, CNN ≈79–80%, MLP ≈53–55%. Per-epoch logs confirm monotonic loss reduction; EfficientNet-Lite0 achieves the best accuracy but at the highest epoch time/energy.
- Accuracy consistency across engines (02): For the same fixed batch, PyTorch, ONNX Runtime (CPU/CUDA), and OpenVINO CPU/GPU produce matching top-1 predictions and near-zero numeric diffs. OpenVINO NPU shows ≈9.375% accuracy for all models, which is inconsistent with the others and indicates a device/kernel layout or post-processing mismatch. Exclude NPU rows from fairness comparisons until resolved.
- Latency and throughput (02):
	- CUDA: ONNX Runtime is consistently faster than PyTorch for CNN/MLP/MobileNetV3; EfficientNet-Lite0 also benefits (e.g., ~5.9 ms vs ~12.0 ms mean latency). OpenVINO on GPU is competitive for MobileNetV3/EfficientNet.
	- CPU: For MobileNetV3, ORT CPU and OpenVINO CPU both outperform PyTorch CPU. For CNN, repeated sessions show variability (one session had ORT CPU ≈3.6 ms; another ≈116 ms). This points to host-level factors (threading, power plan, background load) dominating small models; keep CPU power/affinity fixed for stable results.
- Energy on NVIDIA CUDA (02): NVML-based measurements show ONNX Runtime typically consumes less GPU energy than PyTorch for the same batch/runs (CNN/MLP/MobileNetV3). EfficientNet-Lite0 shows an inversion (PyTorch lower energy despite higher latency), which likely stems from sampling-window and synchronization effects. Treat CUDA energy deltas as directional; repeat with explicit stream sync and sustained clocks for robust conclusions. CPU/iGPU/NPU energy is N/D on Windows by design.
- Reproducibility and fairness: Benchmarks use unified preprocessing, identical fixed batches per run, 10 warmups + 100 timed runs, and write environment metadata (driver/runtime versions, device names, model hash) into each CSV row. Caching prevents accidental duplication. Given the observed CPU variance and NPU mismatch, pin ORT/OpenVINO thread settings and exclude NPU until validated.

Follow-ups
----------
- Investigate OpenVINO NPU accuracy collapse (likely layout or unsupported op fallback); add a dedicated validation cell before timing and block NPU from comparisons until fixed.
- Stabilize CPU runs: pin intra/inter-op threads, set a fixed power plan, and isolate background load; rerun CNN to reconcile the ORT CPU discrepancy seen across sessions.
- Harden energy metering: ensure explicit device sync around timing windows; consider sampling total system draw or per-rail telemetry where available; repeat EfficientNet-Lite0 energy to confirm the anomaly.
