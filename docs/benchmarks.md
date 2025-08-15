# Benchmarks

Scope
- Models: CNN, MLP, MobileNetV3, EfficientNet-Lite0
- Engines: PyTorch, ONNX Runtime, OpenVINO
- Providers: CPU, CUDA (GPU), optional DirectML (ORT), OpenVINO GPU/NPU (if available)
 - Federated simulation: Flower (local simulation of multiple clients)

Configuration
- Centralized matrix in [bench matrix](../config/bench_matrix.yaml) (with notebook fallback defaults)
- Parameters: batch, warmup, runs, image size, workers

Execution details
- 01 notebook trains and saves PyTorch checkpoints and ONNX exports.
- 02 notebook performs consistent inference runs across engines:
  - Identical preprocessed batch is fed to each engine.
  - Energy measured for CUDA paths via NVML; others N/D.
  - Consistency checks compare logits across engines to validate numerical equivalence.
- Results written to metrics/engines_bench.csv (and metrics/inference_energy_summary.csv for energy-only plots) with a stable schema.
- 03 notebook runs a Flower-based FL simulation with per-round/per-client metrics. Results written to metrics/03_flower_rounds.csv.

Interpreting metrics
- lat_ms_mean, lat_ms_p95: lower is better.
- thr_ips: higher is better.
- acc: sanity check on the test batch.
- energy_j: lower indicates better efficiency (when measured).
 - For 03_flower_rounds.csv: t_train_s, t_eval_s, t_agg_s per round; client/server accuracy; bytes_up/down and params_bytes for communication overhead; optional energy_j for CUDA clients; optional ov_* columns for OpenVINO eval micro-benchmarks.

Next: [Results](./results.md)

