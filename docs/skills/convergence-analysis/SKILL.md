# Skill: Convergence Analysis

Refer to this document to learn about TD3/PPO convergence, erratic reward curves, best checkpoints, reward smoothing, auxiliary training metrics, or professor-facing convergence explanations.

## Relevant Files

- `results/<run>/metrics/td3_convergence_history.json`
- `results/<run>/metrics/ppo_convergence_history.json`
- `results/<run>/td3_convergence_training_report.json`
- `results/<run>/ppo_convergence_training_report.json`
- `leader_dt/plotting/convergence_plots.py`
- `scripts/train_td3_until_convergence.py`
- `scripts/train_ppo_until_convergence.py`

## Interpretation Rules

- Jagged reward does not automatically mean failure.
- Reward is penalty-heavy: freshness, accuracy, and terminal CPU penalties create step-like changes.
- Always inspect auxiliary metrics: AoI, freshness violations, accuracy violations, CPU backlog, and penalized score.
- Use moving averages for visual clarity and state the smoothing window in captions.
- Do not claim smooth convergence unless smoothed reward and system metrics support it.
- Use final Monte Carlo as the main performance evidence.

## Replot Convergence Metrics

```bash
python scripts/replot_convergence.py \
  --input-json results/<run>/metrics/td3_convergence_history.json \
  --output-dir results/<run>/plots_replotted
```

If `scripts/replot_convergence.py` is not available, generate plots directly from the JSON with pandas/matplotlib and include raw plus rolling-mean curves.