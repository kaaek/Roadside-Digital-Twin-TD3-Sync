# Skill: Evaluate with Monte Carlo

Refer to this document to compare trained policies, generate final performance metrics, inspect `monte_carlo_metrics.json`, or run a small evaluation smoke test.

## Relevant Files

- `scripts/run_monte_carlo.py`
- `leader_dt/evaluation/policy_factory.py`
- `leader_dt/evaluation/monte_carlo.py`
- `leader_dt/evaluation/rollout.py`
- `leader_dt/evaluation/metrics.py`
- `leader_dt/evaluation/reporting.py`
- `leader_dt/rl/wrappers.py`
- `leader_dt/baselines/greedy.py`

## Final Command Template

```bash
python scripts/run_monte_carlo.py \
  --td3-model-path results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip \
  --ppo-model-path results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip \
  --trials 1000 \
  --seed-start 50000 \
  --output-dir results/final_monte_carlo_td3_ppo_greedy
```

## Smoke Command

```bash
python scripts/run_monte_carlo.py \
  --td3-model-path results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip \
  --ppo-model-path results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip \
  --trials 10 \
  --seed-start 50000 \
  --output-dir results/smoke_monte_carlo_td3_ppo_greedy
```

## Metrics to Report

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

## Policy Set

The final paper-facing policy set should be:

```text
Greedy
TD3
PPO
```
