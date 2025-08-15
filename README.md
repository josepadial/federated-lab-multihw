federated-lab-multihw
======================

Clean, reproducible benchmarking across PyTorch, ONNX Runtime (CPU/CUDA), OpenVINO, and a Flower-based federated learning simulation on Windows with a single virtual environment.

Documentation
-------------
- See [README](docs/README.md) for the full documentation set: overview, getting started, reproducibility, benchmarks (incl. federated simulation), results, energy, troubleshooting.

Notebook flow
-------------
- [00_check_env_and_tests.ipynb](notebooks/00_check_env_and_tests.ipynb) — validate environment, providers, and run smoke tests; writes reports/ artifacts.
- [01_train_and_baseline.ipynb](notebooks/01_train_and_baseline.ipynb) — train on PyTorch (CUDA if available), export ONNX, and record PyTorch CPU/CUDA baselines.
- [02_inference_engines_bench.ipynb](notebooks/02_inference_engines_bench.ipynb) — benchmark PyTorch/ORT/OpenVINO fairly with unified preprocessing and NVML energy (CUDA only); caching avoids duplicate runs. Outputs: `metrics/engines_bench.csv`, `metrics/inference_energy_summary.csv`.
- [03_flower_simulation.ipynb](notebooks/03_flower_simulation.ipynb) — run a local FL simulation with Flower, collect per-round/per-client timing, accuracy, and optional CUDA energy. Outputs: `metrics/03_flower_rounds.csv`.
- [04_results_and_plots.ipynb](notebooks/04_results_and_plots.ipynb) — aggregate CSVs and produce plots/tables under `reports/figs/`.

Windows setup (single venv)
---------------------------
- Install PyTorch and torchvision for your CUDA using the official index URL.
- Install onnxruntime-gpu (provides CPU and CUDA providers) and OpenVINO wheels.
- Install Flower: `flwr[simulation]` (already in `requirements.txt`).
- Optional: nvidia-ml-py3 for NVML-based GPU energy measurement.

Quick start
-----------
1) Run 00 to verify environment and providers; fix any issues before continuing.
2) Run 01 to train chosen models and export ONNX.
3) Run 02 to benchmark across engines; results go to `metrics/engines_bench.csv` with caching.
4) Run 03 to execute the federated learning simulation and write `metrics/03_flower_rounds.csv`.
5) Run 04 to generate plots/tables from the CSV(s).

Notes
-----
- Energy is measured via NVML on NVIDIA GPUs in 02 and optionally within CUDA clients in 03; CPU/iGPU/NPU are reported as "N/D" on Windows.
- OpenVINO devices are detected at runtime; unavailable devices are skipped and noted in the CSV. In 03, OpenVINO inference metrics may be recorded during local evaluation when enabled.

Interpretation (current metrics snapshot)
----------------------------------------
Sources: `metrics/train_baseline.csv`, `metrics/engines_bench.csv`, `metrics/*_train_history.csv`, `metrics/03_flower_rounds.csv`, and `logs/run.log`.

- Training dynamics (01): All models converge on CIFAR-10. Final top-1: EfficientNet-Lite0 ≈86–87%, MobileNetV3 ≈80–81%, CNN ≈79–80%, MLP ≈50–51%.
- Cross-engine (02): PyTorch/ORT/OpenVINO CPU/GPU produce matching top-1 and tiny numeric diffs. CUDA: ORT is typically faster than PyTorch; OpenVINO GPU is competitive for MobileNetV3/EfficientNet. NPU rows execute but current throughput numbers appear inconsistent with reported latencies; treat NPU results as provisional.
- Energy on CUDA (02): ORT generally uses less GPU energy than PyTorch for the same runs (CNN, MobileNetV3, EfficientNet-Lite0). MLP shows a slight exception where PyTorch is marginally lower.
- Federated simulation (03):
	- Scenario A (homogeneous CPU) shows steady round-to-round accuracy gains at clients and server (server acc ≈0.10→0.45 over three aggregations), with ~28–30 s local train and ~17–18 s eval per client.
	- Scenario B (heterogeneous: 1 CUDA + CPUs) adds `energy_j` for the CUDA client (≈203→166→158 J across repeats), indicating decreasing energy per local epoch as caching/warm state stabilize.
	- Scenario C (instrumented eval) records optional OpenVINO eval micro-metrics (ov_* columns) alongside FL metrics and shows a stronger energy decrease on the CUDA client (≈262→142→71 J) as runs warm up.
	- Scenario D stresses convergence (lower local compute or skew), leading to lower accuracies (≈0.16–0.31), as expected.

Project conclusions (executive summary)
--------------------------------------
See full details below; in brief, the project achieved a clean baseline training/benchmarking stack across engines and added a first FL simulation with actionable metrics. Heterogeneous clients are supported, and CSV outputs enable reproducible analysis and plotting.

Conclusions of the project
--------------------------
Objectives
- Build a reproducible, Windows-friendly lab to train, export, benchmark, and simulate FL with small vision models.
- Compare engines fairly (PyTorch, ONNX Runtime, OpenVINO) and collect energy on CUDA.
- Validate a minimal Flower simulation capturing per-round client/server metrics.

What was achieved
- Training converges to expected CIFAR-10 accuracies; artifacts exported to ONNX and OpenVINO IRs.
- Cross-engine benchmarks confirm numerical agreement and predictable performance wins for ORT CUDA over PyTorch.
- CUDA energy recorded for multiple models; results are directionally consistent (ORT ≤ PyTorch), with one inversion to revisit.
- FL simulation produces structured CSV (`03_flower_rounds.csv`) with per-round/per-client times, accuracy, and energy (for CUDA clients); heterogeneous scenarios (CPU+CUDA) are exercised.

Learnings
- Threading and power settings strongly affect CPU latency; pin threads and power plan for stability.
- Energy sampling windows and missing explicit sync can bias per-batch Joules; add barriers for precise attribution.
- NPU paths require deeper validation (layout, ops support) before inclusion in head-to-head comparisons.
- In FL, even simple heterogeneity (CUDA + CPU) alters time/energy trade-offs and fairness of aggregation.

