# Skill: Baselines and CPU-Aware Greedy

## Trigger

Use this skill when the user asks about Greedy, CPU-aware Greedy, baseline fairness, baseline implementation, or why Greedy outperforms TD3.

## Relevant Files

- `leader_dt/baselines/greedy.py`
- `leader_dt/baselines/random_policy.py`
- `leader_dt/baselines/no_refresh.py`
- `leader_dt/constants.py`
- `leader_dt/config.py`
- `scripts/run_monte_carlo.py`
- `scripts/run_sensitivity.py`

## CPU-Aware Greedy Formula

```text
score(i,t) = priority_weight(i) * AoI(i,t)
             - lambda_cpu * max(0, current_backlog + estimated_CPU_cycles(i,t) - slot_CPU_capacity) / slot_CPU_capacity
```

Where:

```text
estimated_CPU_cycles(i,t) = estimated_collected_bits(i,t) * cpu_cycles_per_bit(i)
slot_CPU_capacity = leader_cpu_frequency_cycles_per_second * slot_duration_seconds
```

## Unit Consistency

- `current_backlog`: CPU cycles.
- `estimated_CPU_cycles`: CPU cycles.
- `slot_CPU_capacity`: CPU cycles.
- `max(0, current_backlog + estimated_CPU_cycles - slot_CPU_capacity) / slot_CPU_capacity`: unitless.
- `priority_weight * AoI`: weighted-AoI score.
- `lambda_cpu`: score-space penalty coefficient.

The objective is a heuristic score, not a pure physical equation.

## Defaults

- `DEFAULT_GREEDY_CPU_LAMBDA = 5.0`
- `DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION = 1.0`

Set `lambda_cpu=0` to recover CPU-blind weighted-AoI Greedy behavior.

## Commands

Monte Carlo with CPU-aware Greedy:

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 500 \
  --seed-start 50000 \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

Recover old Greedy behavior:

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 500 \
  --seed-start 50000 \
  --greedy-lambda-cpu 0 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo_greedy_cpu_blind
```

Minimum-accuracy Greedy test:

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 500 \
  --seed-start 50000 \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 0.8 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo_greedy_min_accuracy
```

## Interpretation

- If Greedy has lower AoI/freshness violations but higher CPU backlog, it is prioritizing freshness over CPU.
- If CPU-aware Greedy keeps AoI low while reducing CPU backlog, a simple heuristic may be sufficient.
- If TD3 lowers CPU backlog but has much worse AoI/freshness, describe it as CPU-conservative but freshness-suboptimal.
