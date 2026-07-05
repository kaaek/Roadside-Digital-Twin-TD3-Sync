# Skill: Baselines and CPU-Aware Greedy

Refer to this doc when the user asks about Greedy, CPU-aware Greedy, baseline fairness, why Greedy improves/worsens in sensitivity plots, or how Greedy compares with TD3/PPO.

## Relevant Files

- `leader_dt/baselines/greedy.py`
- `leader_dt/evaluation/policy_factory.py`
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
- Normalized CPU overflow: unitless.
- `priority_weight * AoI`: weighted-AoI score.
- `lambda_cpu`: score-space penalty coefficient.

## Current Baseline Policy

Greedy is the main baseline in final plots. `Random` and `No refresh` are diagnostic/legacy only and should not be included in final paper plots unless explicitly requested.
