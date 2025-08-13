# Benchmarks

Scope
- Models: CNN, MLP, MobileNetV3, EfficientNet-Lite0
- Engines: PyTorch, ONNX Runtime, OpenVINO
- Providers: CPU, CUDA (GPU), optional DirectML (ORT), OpenVINO GPU/NPU (if available)

Configuration
- Centralized matrix in [bench matrix](../config/bench_matrix.yaml) (with notebook fallback defaults)
- Parameters: batch, warmup, runs, image size, workers

Execution details
- 01 notebook trains and saves PyTorch checkpoints and ONNX exports.
- 02 notebook performs consistent inference runs across engines:
  - Identical preprocessed batch is fed to each engine.
  - Energy measured for CUDA paths via NVML; others N/D.
  - Consistency checks compare logits across engines to validate numerical equivalence.
- Results written to metrics/infer_metrics.csv with a stable schema.

Interpreting metrics
- lat_ms_mean, lat_ms_p95: lower is better.
- thr_ips: higher is better.
- acc: sanity check on the test batch.
- energy_j: lower indicates better efficiency (when measured).

Next: [Results](./results.md)

