# Leader-Assisted DT Synchronization for RSU Failure: Simulator

<!--
This package implements the System Model and Problem Formulation from the paper
around a Zone B defective-RSU simulation.


## Key modeling decisions

- The scheduled entity is the exact sensor-vehicle pair `i ∈ I`.
- Freshness and the weighted AoI objective are evaluated at the sensor-type level; pair decisions still determine the provider, wireless capacity, data size, CPU load, and accuracy.
- Available data size `δ_i(t)` is sampled per pair and per slot.
- Vehicles physically move through Zone B every slot.
- Only Zone B is modeled in code.
- The leader vehicle is currently included as a possible sensor provider.
- TD3 receives penalties for freshness, accuracy, and terminal CPU violations;
  no safety override is applied.
- MILP is excluded from the active execution path for now and kept as a future
  interface only.

-->

## Run

```bash
pip install -e .
python scripts/run_smoke_tests.py
```

## Main package layers

- `domain/`: vehicles, sensor types, exact pair indexing, scenario generation.
- `models/`: communication, AoI, CPU, accuracy, and weighted objective equations.
- `simulator/`: state, action decoding, transition dynamics, Gym environment.
- `baselines/`: greedy/no-refresh/random policies.
- `evaluation/`: rollout and Monte Carlo interfaces.
- `rl/`: TD3 wrappers and training helpers.

## TD3 convergence training

Use this script to train TD3 in evaluation-driven chunks instead of stopping at a fixed timestep without checking convergence.
The defaults are centralized in `leader_dt/constants.py` and exposed through `Td3ConvergenceTrainingConfig` in `leader_dt/config.py`.

Default behavior:

- evaluate every 25,000 training steps;
- use 20 evaluation episodes;
- require at least 500,000 training steps before early stopping;
- stop after 10 consecutive evaluations without reward improvement;
- cap training at 3,000,000 steps unless overridden;
- save the best model, latest model, evaluation history, and convergence plots.

Run:

```bash
python scripts/train_td3_until_convergence.py
```

Use a larger training budget:

```bash
python scripts/train_td3_until_convergence.py \
  --maximum-timesteps 5000000 \
  --minimum-timesteps 500000 \
  --eval-frequency-steps 25000 \
  --evaluation-episodes 20 \
  --patience-evaluations 10
```

Outputs are saved under `results/convergence_td3/` by default:

- `models/best_td3_exact_pair_zone_b.zip`
- `models/latest_td3_exact_pair_zone_b.zip`
- `metrics/td3_convergence_history.json`
- `metrics/td3_convergence_history.csv`
- `plots/td3_evaluation_reward_convergence.png`
- `plots/td3_evaluation_aoi_convergence.png`
- `td3_convergence_training_report.json`

Evaluate the best converged model:

```bash
python scripts/run_monte_carlo.py \
  --model-path results/convergence_td3/models/best_td3_exact_pair_zone_b.zip \
  --trials 100
```
