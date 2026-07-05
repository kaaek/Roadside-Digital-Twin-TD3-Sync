# Context: TD3-RSU Leader-Assisted Digital Twin Synchronization

## Purpose

This document for the `td3-rsu` codebase captures the current research objective, package structure, command surface, modeling assumptions, experiment protocol, plotting conventions, and reporting claims.

## Current Project State

`td3-rsu` is a Python research package for leader-assisted vehicular Digital Twin synchronization during RSU failure. A leader vehicle substitutes for the failed RSU by collecting data from nearby vehicles and updating the Digital Twin while balancing freshness, communication bandwidth, CPU processing, and accuracy constraints.

The project now supports two reinforcement-learning models:

- `TD3` through `leader_dt/rl/td3_agent.py`.
- `PPO` through `leader_dt/rl/ppo_agent.py`.

The final paper-facing policy comparison is:

```text
Greedy
TD3
PPO
```

`Random` and `No refresh` may still exist as legacy or diagnostic baselines if present in the repository, but they should not be included in any results. Evaluation policy construction is centralized in `leader_dt/evaluation/policy_factory.py`.

## Research Question

The project asks whether continuous-control reinforcement learning can learn a robust scheduling and accuracy-control policy for leader-assisted Digital Twin updates under stochastic vehicular conditions, and how TD3/PPO compare against a CPU-aware Greedy heuristic across Monte Carlo trials and sensitivity sweeps.

## Core Modeling Invariants

- The defective infrastructure is an RSU covering Zone B.
- The leader vehicle acts as a temporary collector/uploader during the failure interval.
- Vehicles carry subsets of sensor types.
- The action is pair-level: the agent scores padded provider pairs `i = (vehicle, sensor_type)` and outputs one requested accuracy fraction.
- AoI/freshness/objective are sensor-type-level.
- Communication, data availability, accuracy, and CPU feasibility are pair-level.
- One feasible provider pair is scheduled per time slot.
- Refreshing any valid provider pair for sensor type `s` refreshes the Digital Twin state for sensor type `s`.
- Evaluation should use deterministic policy prediction and deterministic action decoding.
- Monte Carlo evaluation changes scenario seeds but keeps policy behavior deterministic within each scenario.

## Main Package Structure

```text
leader_dt/
  constants.py                 # global defaults
  config.py                    # dataclass configuration objects
  domain/                      # scenario, vehicles, sensors, topology, pairs
  models/                      # AoI, accuracy, CPU, communication, objective models
  simulator/                   # Gymnasium environment, action decoding, dynamics, state
  rl/                          # observation, reward, TD3, PPO, SB3 wrappers
  baselines/                   # Greedy and optional legacy diagnostic baselines
  evaluation/                  # rollout, Monte Carlo, sensitivity, metrics, policy factory
  plotting/                    # convergence/comparison/sensitivity plots and thesis style
  verification/                # Z3 feasibility diagnostics
scripts/
  train_td3.py
  train_td3_until_convergence.py
  train_ppo.py
  train_ppo_until_convergence.py
  run_monte_carlo.py
  run_sensitivity.py
  check_z3_feasibility.py
```

## Current CLI Surface

### Compile and test

```bash
python -m compileall -q leader_dt scripts tests
pytest -q
```

### Train TD3

```bash
python scripts/train_td3_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 3000000 \
  --minimum-timesteps 3000000 \
  --eval-frequency-steps 100000 \
  --evaluation-episodes 50 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50 \
  --action-noise-sigma 0.05 \
  --target-policy-noise 0.20 \
  --target-noise-clip 0.30 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --checkpoint-frequency-steps 250000 \
  --device cuda \
  --output-dir results/td3_zone_2km_3m_seed1_gpu \
  --tensorboard-log-dir results/td3_zone_2km_3m_seed1_gpu/tensorboard \
  --monitor-log-dir results/td3_zone_2km_3m_seed1_gpu/monitor
```

### Train PPO

```bash
python scripts/train_ppo_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 3000000 \
  --minimum-timesteps 3000000 \
  --eval-frequency-steps 100000 \
  --evaluation-episodes 50 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50 \
  --learning-rate 0.0003 \
  --n-steps 2048 \
  --batch-size 256 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-range 0.20 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --checkpoint-frequency-steps 250000 \
  --device cpu \
  --output-dir results/ppo_zone_2km_3m_seed1_cpu \
  --tensorboard-log-dir results/ppo_zone_2km_3m_seed1_cpu/tensorboard \
  --monitor-log-dir results/ppo_zone_2km_3m_seed1_cpu/monitor
```

PPO with MLP policies is usually CPU-efficient; Stable-Baselines3 may warn that GPU utilization is poor for PPO unless using CNN policies. TD3 can benefit from GPU if local benchmarking shows a meaningful speedup.

### Monte Carlo evaluation with TD3 and PPO

```bash
python scripts/run_monte_carlo.py \
  --td3-model-path results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip \
  --ppo-model-path results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip \
  --trials 1000 \
  --seed-start 50000 \
  --output-dir results/final_monte_carlo_td3_ppo_greedy
```

`--model-path` remains a backward-compatible TD3 alias if present in the current source, but prefer `--td3-model-path` and `--ppo-model-path` in new commands.

### Sensitivity sweep with TD3 and PPO

```bash
python scripts/run_sensitivity.py \
  --parameter data_size_high_multiplier \
  --values 0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.3,2.6,3.0,3.4,3.8,4.2,4.8,5.4,6.0,6.8,7.6,8.4,9.2 \
  --trials 500 \
  --seed-start 50000 \
  --td3-model-path results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip \
  --ppo-model-path results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip \
  --output-dir results/sensitivity_final_task_size_20_values
```

`leader_dt/plotting/sensitivity_plots.py` should enforce the thesis plotting style by default. Sensitivity plots should save both `.png` and `.pdf` whenever possible.

## Current Results and Claims

Current sensitivity results indicate that TD3 is comparatively stable across task size, sensors-per-vehicle, and accuracy-threshold sweeps. PPO demonstrates even more promising results than TD3 at scale. Greedy shows stronger sensitivity in several sweeps, especially as accuracy requirements or task pressure increase. Current claims:

- TD3 and PPO are both implemented as RL policies over the same environment, observation, action semantics, and reward.
- The final fair comparison should use the same Monte Carlo seed range for Greedy, TD3, and PPO.
- Sensitivity sweeps test generalization of fixed trained models; they should not retrain TD3/PPO at each sensitivity point unless the experiment explicitly studies retraining.
- A jagged reward curve is expected because the reward uses discontinuous penalties; conclusions should use reward plus AoI, freshness violations, accuracy violations, CPU backlog, and Monte Carlo statistics.
- Preliminary TD3 sensitivity plots suggest TD3 is less sensitive than Greedy to several system-parameter changes, but final claims should be based on cleaned plots using only Greedy, TD3, and PPO.
