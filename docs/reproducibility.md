# Reproducibility and Methodology

Design principles
- Deterministic flows where feasible (dataset shuffles, initializations, seeds).
- Single-venv install with pinned versions captured in requirements.txt.
- All metrics written to CSVs with full contextual columns (engine, provider, batch, warmup, runs, model hash, versions).
- Caching avoids duplicate work and preserves history across reruns.

Data handling
- CIFAR-10 test set loaded consistently; preprocessing normalized to NCHW float32 with dataset stats.
- Shared preprocess_np() used by all engines to ensure fairness.

Benchmarking protocol
- Warmup and run counts fixed by config; identical batch used across engines for a given benchmark.
- Latency: per-inference time mean and p95; Throughput: items/sec = total items / total elapsed.
- Accuracy: top-1 over the batch if labels are available.
- Energy: measured on NVIDIA GPUs only via NVML, reported in Joules per run segment; others marked N/D.

Traceability
- Model hashes (SHA-256) recorded; driver/runtime versions captured with OS info.
- Providers availability stored in reports/providers_status.csv; environment report saved from 00.

Next: [Benchmarks](./benchmarks.md)
