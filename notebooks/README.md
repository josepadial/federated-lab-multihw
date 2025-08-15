# Notebooks

Execution order and purpose
- 00: Validate environment and providers; generate quick reports.
- 01: Train baselines (PyTorch), export ONNX, log metrics.
- 02: Benchmark inference across engines/providers; cache results; optional energy measurement.
- 03: Flower-based federated learning simulation; logs per-round/per-client metrics to metrics/03_flower_rounds.csv (with optional CUDA energy for GPU clients).
- 04: Aggregate metrics and generate publication-ready figures.

Outputs
- models_saved/*: artifacts from training and conversion
- metrics/engines_bench.csv, metrics/inference_energy_summary.csv, metrics/03_flower_rounds.csv
- reports/*: environment and provider summaries, quick checks, and figures

Navigate back: [README](../docs/README.md)
