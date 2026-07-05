# Skill: Run Sensitivity Sweeps

Refer to this document to perform robustness tests, parameter sweeps, sensitivity plots, thesis-style sensitivity figures, or commands for specific environment-parameter sweeps.

## Relevant Files

- `scripts/run_sensitivity.py`
- `leader_dt/evaluation/policy_factory.py`
- `leader_dt/evaluation/sensitivity.py`
- `leader_dt/plotting/sensitivity_plots.py`
- `leader_dt/plotting/thesis_style.py`
- `leader_dt/config.py`
- `leader_dt/constants.py`

## Default Policy Set

Use:

```text
Greedy
TD3
PPO
```

## Thesis Plotting Requirement

Sensitivity plots must use the project thesis style by default through `leader_dt/plotting/sensitivity_plots.py`. The user should not need a CLI flag. Plots should use SciencePlots, grid styling, serif fonts, accessible colors, distinct markers/linestyles, tight bounding boxes, and PDF export.

## Model Path Setup

```bash
TD3_MODEL="results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip"
PPO_MODEL="results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip"
TRIALS=500
SEED_START=50000
```

## Task Size Sweep

`data_size_high_multiplier` must be greater than or equal to the configured low multiplier. If the default low multiplier is `0.8`, do not sweep below `0.8`.

```bash
python scripts/run_sensitivity.py \
  --parameter data_size_high_multiplier \
  --values 0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.3,2.6,3.0,3.4,3.8,4.2,4.8,5.4,6.0,6.8,7.6,8.4,9.2 \
  --trials "$TRIALS" \
  --seed-start "$SEED_START" \
  --td3-model-path "$TD3_MODEL" \
  --ppo-model-path "$PPO_MODEL" \
  --output-dir results/sensitivity_final_task_size_20_values
```

## Sensors Per Vehicle Sweep

Use integer values only. The valid range depends on the maximum sensors-per-vehicle and number of sensor types configured in the action/observation space.

```bash
python scripts/run_sensitivity.py \
  --parameter sensors_per_vehicle \
  --values 1,2,3,4,5,6,7,8 \
  --trials "$TRIALS" \
  --seed-start "$SEED_START" \
  --td3-model-path "$TD3_MODEL" \
  --ppo-model-path "$PPO_MODEL" \
  --output-dir results/sensitivity_final_sensors_per_vehicle
```

## Accuracy Threshold Sweep

```bash
python scripts/run_sensitivity.py \
  --parameter accuracy_threshold \
  --values 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00 \
  --trials "$TRIALS" \
  --seed-start "$SEED_START" \
  --td3-model-path "$TD3_MODEL" \
  --ppo-model-path "$PPO_MODEL" \
  --output-dir results/sensitivity_final_accuracy_threshold_20_values
```

## Rules

- Sweep one parameter at a time.
- Keep TD3/PPO model paths fixed across values.
- Use the same seed range for all policies and parameter values.
- Do not retrain inside a sensitivity sweep unless explicitly studying retraining.
- Plot additional metrics such as freshness violations, accuracy violations, and CPU backlog when AoI saturates or appears flat.
