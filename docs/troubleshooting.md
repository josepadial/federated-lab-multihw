# Troubleshooting

Common issues
- Missing CUDA provider in ONNX Runtime: ensure onnxruntime-gpu is installed and driver is up-to-date.
- OpenVINO GPU unavailable: verify Intel Graphics drivers and supported hardware.
- NVML errors: install nvidia-ml-py3, confirm nvidia-smi works, and that the GPU supports energy counters.

CSV not updating
- Check that metrics/ and reports/ are writable (see 00 notebook write permission checks).
- Verify cache keys: if inputs are unchanged, cached rows will be written (latency fields empty by design).

Re-running with changes
- Change warmup/runs/providers or model hash (e.g., retrain/export) to generate new non-cached rows.

Contact
- Create an issue with logs from logs/run.log and a short repro. Include providers_status.csv.

Navigate back: [README](./README.md)
