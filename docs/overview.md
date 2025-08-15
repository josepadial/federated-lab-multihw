# Overview

This repository delivers a rigorous, Windows-oriented framework to benchmark deep-learning inference across multiple engines and run a federated learning simulation:
- PyTorch (CPU/CUDA)
- ONNX Runtime (CPU/CUDA; DirectML optional)
- OpenVINO (CPU/GPU/NPU)
- Flower (local simulation)

Goals
- Fair comparisons with a unified preprocessing pipeline and fixed batching.
- Reproducibility: deterministic seeds, single-venv setup, pinned versions, cached results.
- Robustness: structured logging, graceful degradation when providers/devices aren’t available.
- Evidence for research: consolidated CSVs, publication-ready plots, and traceable artifacts.

Artifacts
- Trained models: models_saved/pytorch/*.pt
- Exchange format: models_saved/onnx/*.onnx
- OpenVINO IR: models_saved/openvino_ir/*.xml
- Metrics CSVs: metrics/*.csv
- FL rounds: metrics/03_flower_rounds.csv
- Reports: reports/* (providers status, environment reports, quickchecks)

Notebook flow (executive summary)
- 00: Environment validation, providers availability, smoke tests; writes a short report.
- 01: Training baselines in PyTorch and exporting ONNX.
- 02: Cross-engine inference benchmarking with caching and optional energy (NVML on NVIDIA).
- 03: Flower-based federated learning simulation with per-round metrics.
- 04: Aggregation and visualization of metrics (latency, throughput, accuracy, energy, FL rounds).

Navigate next: [Getting started](./getting-started.md) • [Flower Simulation (03)](./flower-simulation.md)
