# Results and Figures

Artifacts
- metrics/engines_bench.csv: consolidated inference metrics across engines/providers.
- metrics/inference_energy_summary.csv: filtered energy rows for quick plotting.
- metrics/03_flower_rounds.csv: federated-learning per-round/per-client metrics.
- reports/figs: generated plots from notebook 04 (latency, throughput, accuracy, energy, FL rounds).

Reading the CSV
- engines_bench.csv: each row includes contextual fields (engine, provider, versions, device names) enabling grouped analysis; use pandas groupby to compare mean latency/throughput per model and engine.
- 03_flower_rounds.csv: group by scenario/round/role to chart accuracy vs. round at server, client-time distributions, and cumulative communication (bytes_up/down).

Plot highlights (from 04 notebook)
- Combined Loss/Accuracy vs Epoch overlays per model (from training histories).
- Total training energy per model bar plot (if available).
- Inference latency/throughput comparisons per engine/provider.
- Energy efficiency charts for CUDA where NVML is supported.
- Federated learning: server accuracy vs. round, per-client t_train_s and t_eval_s violin/box plots, energy per CUDA client vs. round.

Reproducing figures
- Re-run notebook 04 after updating metrics; figures are saved to reports/figs.

Next: [Energy](./energy.md)
