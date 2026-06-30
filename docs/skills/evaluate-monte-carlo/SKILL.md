# Skill: Evaluate with Monte Carlo

## Trigger

Use this skill when the user asks to compare TD3 against baselines, evaluate a trained model, generate final metrics, summarize policy performance, or inspect `monte_carlo_metrics.json`.

## Relevant Files

- `scripts/run_monte_carlo.py`
- `leader_dt/evaluation/monte_carlo.py`
- `leader_dt/evaluation/rollout.py`
- `leader_dt/evaluation/metrics.py`
- `leader_dt/evaluation/reporting.py`
- `leader_dt/evaluation/penalized_objective.py`
- `leader_dt/baselines/greedy.py`
- `leader_dt/baselines/random_policy.py`
- `leader_dt/baselines/no_refresh.py`
- `leader_dt/rl/wrappers.py`

## Command Template

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 500 \
  --seed-start 50000 \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

## Smoke Command

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_smoke_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 100 \
  --seed-start 50000 \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_smoke_sigma005_seed1/final_monte_carlo
```

## Metrics to Report

Always inspect and report:

- `average_weighted_aoi_float`
- `maximum_aoi_float`
- `freshness_violation_count_integer`
- `accuracy_violation_count_integer`
- `terminal_cpu_violation_count_integer`
- `final_cpu_backlog_cycles_float`
- `total_collected_bits_float`
- `mean_accuracy_float`
- `episode_return_float`
- `penalized_score_float`

## Output Files

- `monte_carlo_metrics.json`
- `monte_carlo_metrics.csv`
- Any generated comparison plots or reports in the chosen `--output-dir`.

## Interpretation Rules

- Lower AoI is better.
- Fewer freshness violations are better.
- Fewer terminal CPU violations are better.
- Lower final CPU backlog is better.
- Higher episode return is better when comparing reward.
- Lower penalized score is better if the project’s penalized objective is formulated as a cost.
- Do not call TD3 better overall if it only improves CPU backlog but loses badly on AoI/freshness; state the tradeoff.

## Common Pitfalls

- If CLI flags `--greedy-lambda-cpu` or `--greedy-requested-accuracy-fraction` are unrecognized, the local script is older than the CPU-aware Greedy version.
- If model loading fails, verify the `.zip` suffix and the exact output directory from training.
- Use the same `--seed-start` and `--trials` across policies when making claims.
