# Context: TD3-RSU Leader-Assisted Digital Twin Synchronization

## Purpose

This file is always-loaded project context for agents working on the `td3-rsu` codebase. It captures the project overview, structure, dependencies, command surface, modeling assumptions, and output conventions needed to safely modify, run, and document the repository.

## Project Overview

`td3-rsu` is a Python research codebase for a leader-assisted vehicular Digital Twin synchronization problem under RSU failure. A leader vehicle collects sensor data from nearby vehicles and uploads updates to keep Digital Twin sensor-type freshness low while respecting communication, CPU, accuracy, and finite-horizon constraints.

The main research question is whether a TD3 reinforcement-learning policy can outperform or complement heuristic baselines, especially Greedy scheduling, under stochastic vehicular conditions. The current evidence indicates TD3 learns a non-random, CPU-aware policy, but Greedy remains stronger on AoI/freshness under the tested formulation.

## Core Modeling Context

- The failed infrastructure is an RSU covering a defective zone.
- The leader vehicle acts as a temporary collector/uploader during the RSU failure window.
- Vehicles carry subsets of sensor types.
- Actions are pair-level: the scheduler selects a specific provider pair `i = (vehicle, sensor_type)`.
- AoI/freshness/objective are sensor-type-level, not pair-level.
- Communication/data/accuracy/CPU feasibility are pair-level.
- One pair is scheduled per time slot.
- Default horizon is `40` slots with slot duration `1.0` second.
- Default vehicle count is `40`.
- Default sensor type count is `8`.
- Default sensors per vehicle is `4`.
- Default freshness threshold is `10` slots.
- Default accuracy threshold is `0.8`.
- Default leader CPU frequency is `1.5e6` cycles/second.
- Default uplink bandwidth is `600000` Hz.

## Project Structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── leader_dt/
│   ├── constants.py
│   ├── config.py
│   ├── types.py
│   ├── domain/
│   │   ├── entities.py
│   │   ├── scenario.py
│   │   ├── sensor_pairs.py
│   │   └── topology.py
│   ├── models/
│   │   ├── accuracy.py
│   │   ├── aoi.py
│   │   ├── communication.py
│   │   ├── cpu.py
│   │   └── objective.py
│   ├── simulator/
│   │   ├── action.py
│   │   ├── dynamics.py
│   │   ├── environment.py
│   │   ├── recorder.py
│   │   └── state.py
│   ├── rl/
│   │   ├── observation.py
│   │   ├── reward.py
│   │   ├── td3_agent.py
│   │   └── wrappers.py
│   ├── baselines/
│   │   ├── greedy.py
│   │   ├── random_policy.py
│   │   └── no_refresh.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── monte_carlo.py
│   │   ├── penalized_objective.py
│   │   ├── reporting.py
│   │   ├── rollout.py
│   │   └── sensitivity.py
│   ├── optimization/
│   │   ├── milp_constraints.py
│   │   ├── milp_objective.py
│   │   ├── milp_solver.py
│   │   └── milp_variables.py
│   ├── plotting/
│   │   ├── comparison_plots.py
│   │   ├── convergence_plots.py
│   │   └── sensitivity_plots.py
│   ├── io/
│   │   ├── config_io.py
│   │   └── results_io.py
│   └── verification/
│       └── z3_feasibility.py
├── scripts/
│   ├── train_td3.py
│   ├── train_td3_multiseed.py
│   ├── train_td3_until_convergence.py
│   ├── run_monte_carlo.py
│   ├── run_sensitivity.py
│   ├── check_z3_feasibility.py
│   ├── diagnose_policy.py
│   ├── generate_figures.py
│   ├── plot_penalized_comparison.py
│   └── run_smoke_tests.py
└── tests/
    ├── test_environment.py
    ├── test_penalized_objective.py
    └── test_z3_feasibility.py
```

## Dependencies and Software Versions

The project requires Python `>=3.10`.

Declared dependencies:

```text
numpy>=1.24
gymnasium>=0.29
stable-baselines3>=2.3
torch>=2.0
matplotlib>=3.7
tensorboard>=2.14
z3-solver>=4.12
pytest>=8.0
```

Installation command:

```bash
pip install -r requirements.txt
```

Package metadata is defined in `pyproject.toml` under project name `leader-dt`, version `0.2.0`.

## Tool Definitions: Main Commands

### Compile and Smoke-Check

```bash
python -m compileall -q leader_dt scripts tests
pytest -q
python scripts/run_smoke_tests.py
```

### Train TD3

```bash
python scripts/train_td3.py
```

### Train TD3 with Convergence Tracking

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
  --output-dir results/td3_tuned_3m_sigma005_seed1 \
  --tensorboard-log-dir results/td3_tuned_3m_sigma005_seed1/tensorboard \
  --monitor-log-dir results/td3_tuned_3m_sigma005_seed1/monitor
```

### Run Monte Carlo Evaluation

```bash
python scripts/run_monte_carlo.py \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --trials 500 \
  --seed-start 50000 \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

### Run Sensitivity Sweep

```bash
python scripts/run_sensitivity.py \
  --parameter data_size_high_multiplier \
  --values 1.0,1.5,2.0,3.0 \
  --trials 500 \
  --seed-start 50000 \
  --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip \
  --greedy-lambda-cpu 5 \
  --greedy-requested-accuracy-fraction 1.0 \
  --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_data_pressure
```

### Run Z3 Feasibility Diagnostics

```bash
python scripts/check_z3_feasibility.py \
  --vehicle-count 40 \
  --sensors-per-vehicle 4 \
  --time-horizon-slots 40 \
  --freshness-threshold-slots 10 \
  --timeout-ms 60000
```

### View TensorBoard

```bash
tensorboard --logdir results/td3_tuned_3m_sigma005_seed1/tensorboard
```

## Key Configuration Objects

Defined in `leader_dt/config.py`:

- `SystemConfig`: time horizon, slot duration, vehicle count, sensor count, freshness threshold, accuracy threshold, CPU frequency, max padded action dimensions.
- `CommunicationConfig`: uplink bandwidth, transmit power, noise density, pathloss exponent, reference distance.
- `RoadConfig`: lane length, defective-zone interval, vehicle speed, speed jitter.
- `DataGenerationConfig`: data-size low/high multipliers.
- `SimulationConfig`: wrapper combining system, communication, road, data-generation, and random seed.
- `Td3TrainingConfig`: TD3 hyperparameters, action noise, target smoothing, TensorBoard, Monitor, checkpoint settings.
- `Td3ConvergenceTrainingConfig`: evaluation frequency, evaluation episodes, patience, min/max timesteps, output paths, best/latest model names.
- `MonteCarloConfig`: trial count and seed start.
- `SensitivityConfig`: trial count per sensitivity point.

## Key Constants

Defined in `leader_dt/constants.py`:

- `DEFAULT_TIME_HORIZON_SLOTS = 40`
- `DEFAULT_SLOT_DURATION_SECONDS = 1.0`
- `DEFAULT_VEHICLE_COUNT = 40`
- `DEFAULT_SENSOR_TYPE_COUNT = 8`
- `DEFAULT_SENSORS_PER_VEHICLE = 4`
- `DEFAULT_FRESHNESS_THRESHOLD_SLOTS = 10`
- `DEFAULT_ACCURACY_THRESHOLD = 0.80`
- `DEFAULT_UPLINK_BANDWIDTH_HZ = 6.0e5`
- `DEFAULT_LEADER_CPU_FREQUENCY_CYCLES_PER_SECOND = 1.5e6`
- `DEFAULT_BUFFER_SIZE = 1_000_000`
- `DEFAULT_BATCH_SIZE = 256`
- `DEFAULT_ACTION_NOISE_SIGMA = 0.10`
- `DEFAULT_TARGET_POLICY_NOISE = 0.20`
- `DEFAULT_TARGET_NOISE_CLIP = 0.30`
- `DEFAULT_GREEDY_CPU_LAMBDA = 5.0`
- `DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION = 1.0`

## Sensor Definitions

Default sensor definitions are ordered by `sensor_type_id`:

| Sensor | Priority Weight | CPU Cycles/Bit | Nominal Bits |
|---|---:|---:|---:|
| Front Camera | 3.0 | 15.0 | 300000 |
| Radar | 2.2 | 2.0 | 80000 |
| Engine Temperature | 2.0 | 3.0 | 60000 |
| Battery BMS | 1.8 | 2.0 | 50000 |
| Tyre Pressure TPMS | 1.5 | 8.0 | 200000 |
| IMU Accelerometer | 1.3 | 11.0 | 220000 |
| Ambient Weather | 1.2 | 10.0 | 250000 |
| Fuel Level | 1.0 | 1.0 | 40000 |

## TD3 Action and Observation Context

The TD3 policy is implemented with Stable-Baselines3 `TD3` in `leader_dt/rl/td3_agent.py`.

Action structure:

- One continuous score per padded provider-sensor pair.
- One continuous requested accuracy fraction.
- The simulator decodes the score vector into a feasible scheduled pair.

Observation structure:

- Normalized pair/sensor freshness features.
- Feasibility information.
- Available data information.
- CPU backlog and previous CPU load.
- Time progress and urgency features.

Do not assume the action is sensor-type-only. In the package version, TD3 selects exact pair-level scores.

## Baselines

Baseline policies live in `leader_dt/baselines/`:

- `GreedyWeightedAoiPolicy`: CPU-aware weighted AoI Greedy baseline.
- `GreedyMaxAoiPolicy`: max-AoI-style Greedy policy when present/used.
- `RandomPolicy`: random feasible scheduling.
- `NoRefreshPolicy`: no scheduled refresh baseline.

CPU-aware Greedy score:

```text
score(i,t) = priority_weight(i) * AoI(i,t)
             - lambda_cpu * max(0, current_backlog + estimated_CPU_cycles(i,t) - slot_CPU_capacity) / slot_CPU_capacity
```

CPU units:

- `current_backlog`: CPU cycles.
- `estimated_CPU_cycles`: CPU cycles.
- `slot_CPU_capacity`: CPU cycles.
- normalized backlog term: unitless.
- `lambda_cpu`: score-space coefficient in weighted-AoI penalty units per normalized CPU slot.

## Evaluation Metrics

Common metrics include:

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

Monte Carlo outputs usually include:

- `monte_carlo_metrics.json`
- `monte_carlo_metrics.csv`
- plots/reports depending on script options.

Convergence outputs usually include:

- `metrics/td3_convergence_history.json`
- `metrics/td3_convergence_history.csv`
- `td3_convergence_training_report.json`
- `plots/td3_evaluation_reward_convergence.png`
- `plots/td3_evaluation_aoi_convergence.png`
- `models/best_td3_exact_pair_zone_b.zip`
- `models/latest_td3_exact_pair_zone_b.zip`
- `checkpoints/td3_checkpoint_*_steps.zip`
- `tensorboard/`
- `monitor/`

## Sensitivity Parameters

Known useful sensitivity parameters:

- `vehicle_count`
- `zone_size_meter`
- `freshness_threshold_slots`
- `data_size_high_multiplier`
- `vehicle_speed_meter_per_second`
- `sensors_per_vehicle`
- `uplink_bandwidth_hz`
- `accuracy_threshold`
- `time_horizon_slots`

Greedy-specific parameters such as `--greedy-lambda-cpu` and `--greedy-requested-accuracy-fraction` should be treated as baseline configuration, not environment sensitivity parameters.

## Experimental Notes

- The previous nuclear run used `5` seeds and `10,000,000` timesteps per seed for `50,000,000` total TD3 training timesteps.
- The nuclear run did not produce a clean visual reward plateau.
- Best checkpoints often appeared before the final timestep, indicating non-monotonic training.
- TD3 generally reduced CPU backlog relative to Greedy but did not beat Greedy on AoI/freshness in the main experiments.
- Reward curves are expected to be noisy because the reward is penalty-heavy and the environment is stochastic.
- Do not claim TD3 “converged successfully” unless supported by convergence JSON, smoothed plots, and Monte Carlo results.

## Output and Artifact Conventions

Use run-specific output directories under `results/`:

```text
results/<experiment_name>/models/
results/<experiment_name>/metrics/
results/<experiment_name>/plots/
results/<experiment_name>/checkpoints/
results/<experiment_name>/tensorboard/
results/<experiment_name>/monitor/
results/<experiment_name>/final_monte_carlo/
results/<experiment_name>/sensitivity_<parameter>/
```

Prefer descriptive experiment names such as:

```text
td3_tuned_3m_sigma005_seed1
nuclear_td3/seed_1
cpu_aware_greedy_lambda5_acc10
sensitivity_bandwidth_seed5
```
