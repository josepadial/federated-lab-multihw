# Notebook 03: Flower Federated Simulation

Purpose
- Run a local, reproducible FL simulation with heterogeneous clients (CPU and optional CUDA) using Flower.
- Capture per-round/per-client metrics: training/eval times, accuracy, aggregation time, communication bytes, and optional CUDA energy.

Scenarios covered (as implemented in the notebook)
- A: Homogeneous CPU clients (sanity/convergence check).
- B: Heterogeneous mix with one CUDA client (adds `energy_j`).
- C: Instrumented evaluation paths (adds optional OpenVINO eval micro-metrics `ov_*`).
- D: Stress convergence (e.g., fewer local epochs, skewed data) to observe accuracy impacts.

Outputs
- metrics/03_flower_rounds.csv with columns:
  - timestamp, scenario, round, role (server|client), cid, device (cpu|cuda),
  - t_train_s, t_eval_s, t_agg_s,
  - params_bytes, bytes_up, bytes_down,
  - loss, acc, energy_j,
  - ov_cpu_lat_ms, ov_cpu_thr_ips, ov_gpu_lat_ms, ov_gpu_thr_ips, ov_npu_lat_ms, ov_npu_thr_ips,
  - torch, flwr, openvino, os, hostname.

Dependencies
- `flwr[simulation]` (already listed in requirements.txt)
- Optional `nvidia-ml-py3`/`pynvml` for CUDA energy on NVIDIA GPUs

How to run
- Execute the cells in order. Adjust the number of clients, local epochs, and scenario flags at the top of the notebook.
- For reliable timings/energy on CUDA, keep the GPU otherwise idle during runs.

Notes and caveats
- Energy is N/D for non-CUDA clients on Windows.
- Aggregation time (t_agg_s) is near-zero in in-process simulations; grows if using networked/server mode.
- Treat OpenVINO NPU micro-metrics as provisional until validated numerically.

Navigate back: [Benchmarks](./benchmarks.md) | [Results](./results.md)
