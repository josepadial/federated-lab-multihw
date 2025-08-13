# Results and Figures

Artifacts
- metrics/infer_metrics.csv: consolidated inference metrics across engines/providers.
- reports/figs: generated plots from notebook 04 (latency, throughput, accuracy, energy).
- metrics/inference_energy_summary.csv: filtered energy rows for quick plotting.

Reading the CSV
- Each row includes contextual fields (engine, provider, versions, device names) enabling grouped analysis.
- Use pandas groupby to compare mean latency/throughput per model and engine.

Plot highlights (from 04 notebook)
- Combined Loss/Accuracy vs Epoch overlays per model (from training histories).
- Total training energy per model bar plot (if available).
- Inference latency/throughput comparisons per engine/provider.
- Energy efficiency charts for CUDA where NVML is supported.

Reproducing figures
- Re-run notebook 04 after updating metrics; figures are saved to reports/figs.

Next: [Energy](./energy.md)
