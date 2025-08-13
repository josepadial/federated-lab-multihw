# Notebooks

Execution order and purpose
- 00: Validate environment and providers; generate quick reports.
- 01: Train baselines (PyTorch), export ONNX, log metrics.
- 02: Benchmark inference across engines/providers; cache results; optional energy measurement.
- 03: Placeholder for federated learning simulation (future work).
- 04: Aggregate metrics and generate publication-ready figures.

Outputs
- models_saved/*: artifacts from training and conversion
- metrics/*.csv: consolidated measurements
- reports/*: environment and provider summaries, quick checks

Navigate back: [README](../docs/README.md)
