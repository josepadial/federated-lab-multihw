# Energy Measurement

Method
- NVIDIA GPUs only via NVML (nvidia-ml-py3), measured around isolated inference calls.
- Per-run measurement returns energy in Joules and time elapsed; aggregated across runs where needed.
 - In Notebook 03 (Flower), optional NVML sampling can be enabled for CUDA clients during local evaluation/training windows; non-CUDA clients report N/D.

Limitations
- No CPU/iGPU RAPL on Windows; those entries are N/D.
- OpenVINO pathways report N/D unless paired with NVML-capable NVIDIA GPU.

Best practices
- Ensure minimal background GPU load when measuring.
- Synchronize CUDA (torch.cuda.synchronize) around timed sections for accurate boundaries.
- Consider multiple seeds/batches if you need statistical confidence intervals.

Reporting
- The 02 notebook emits inference_energy_summary.csv for quick energy-only analysis.

Return: [main README](../README.md)
