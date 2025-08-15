# Troubleshooting

Common issues
- Missing CUDA provider in ONNX Runtime: ensure onnxruntime-gpu is installed and driver is up-to-date.
- OpenVINO GPU unavailable: verify Intel Graphics drivers and supported hardware.
- NVML errors: install nvidia-ml-py3, confirm nvidia-smi works, and that the GPU supports energy counters.
- Flower simulation stalls or exits early: lower the number of clients or rounds; ensure ports are free if using a networked strategy (simulation uses in-process by default).

CSV not updating
- Check that metrics/ and reports/ are writable (see 00 notebook write permission checks).
- Verify cache keys: if inputs are unchanged, cached rows will be written (latency fields empty by design).
- Confirm correct filenames: engines_bench.csv for 02; 03_flower_rounds.csv for 03.

Re-running with changes
- Change warmup/runs/providers or model hash (e.g., retrain/export) to generate new non-cached rows.
 - For FL, change scenario parameters (clients, local epochs, strategy settings) to produce a new set of rounds in 03_flower_rounds.csv.

Contact
- Create an issue with logs from logs/run.log and a short repro. Include providers_status.csv.

Navigate back: [README](./README.md)
