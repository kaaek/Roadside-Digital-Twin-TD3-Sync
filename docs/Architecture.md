# Architecture Explanation: How the Code Works

## 1. One-sentence summary

The codebase implements a Gymnasium-based vehicular Digital Twin simulator plus TD3/PPO training and evaluation processes for choosing which vehicle-sensor pair to refresh and how much accuracy/data to request.

## 2. High-level approach and main components

The package is organized around a clean separation between simulation, learning, evaluation, and plotting. The simulator creates a stochastic vehicular scenario, tracks the state of the Digital Twin, decodes RL or heuristic actions, updates AoI/CPU/accuracy after each time slot, and returns a reward. The RL modules build TD3 and PPO models on top of the same observation and action spaces. The evaluation modules load trained models and baselines through a policy factory, then run Monte Carlo trials or sensitivity sweeps using identical seed ranges. Plotting modules turn saved metrics into publication-style figures using a mandatory thesis plotting style.

The core state lives in the environment and simulator state objects. It changes once per time slot after a policy selects an action. Observations are read from this simulator state and exposed to TD3/PPO as fixed-size normalized vectors. Rewards are computed after the transition and used by the RL algorithms to update neural-network policies during training.

## 3. Block-by-block explanation

### Configuration and constants

`leader_dt/constants.py` stores default values such as horizon length, vehicle count, sensor count, bandwidth, CPU capacity, TD3/PPO hyperparameters, checkpoint frequencies, Greedy parameters, and plotting defaults. `leader_dt/config.py` wraps these defaults in dataclasses such as `SimulationConfig`, `Td3TrainingConfig`, and `PpoTrainingConfig` so scripts can override values cleanly through CLI arguments.

### Scenario generation

`leader_dt/domain/scenario.py` and related domain modules generate vehicles, sensor ownership, provider-sensor pairs, per-slot sample sizes, Poisson sample-arrival counts, and mobility-related conditions. This is where stochastic scenario information is created from controlled random seeds. Arrival counts are drawn after sample sizes, so a given seed produces the same sizes for every arrival rate. With the arrival rate set to `None`, every pair receives a sample in every slot (the legacy model).

### Simulator environment

`leader_dt/simulator/environment.py` exposes the Gymnasium environment. It stores the current simulation config, scenario, simulator state, observation builder, reward calculator, and action decoder. The environment receives a continuous action vector, decodes it into a scheduled pair plus requested accuracy, applies dynamics, computes reward, and advances time.

### Action decoding

`leader_dt/simulator/action.py` converts RL action vectors into physical scheduling decisions. The action contains one score for every padded provider pair plus one requested accuracy fraction. During evaluation, decoding should select the feasible pair with the highest score deterministically.

### Dynamics and state transition

`leader_dt/simulator/dynamics.py` updates vehicle feasibility, pending sensor samples, sensor-type AoI, achieved accuracy, collected bits, CPU backlog, and violations after each scheduled upload. It enforces the difference between sensor-type freshness and pair-level resource constraints. Each pair holds at most one pending sample in `SimulationState`: new arrivals overwrite it and an upload consumes it. A pair is feasible only if its carrier is in Zone B and it holds a pending sample. A successful refresh sets the sensor-type AoI to the sample's age plus sensing and transmission delay.

### Observation and reward

`leader_dt/rl/observation.py` builds the normalized observation vector consumed by TD3/PPO: per-pair AoI, feasibility, pending sample size and (when the arrival process is enabled) pending sample age, plus four global features. `leader_dt/rl/reward.py` computes a smooth shaped reward from normalized weighted AoI, squared freshness and accuracy slack, a log1p CPU-backlog term, a terminal CPU term, and an accuracy bonus. Reported evaluation metrics stay count-based.

### TD3 and PPO trainers

`leader_dt/rl/td3_agent.py` builds Stable-Baselines3 TD3 with replay buffer, twin critics, target policy smoothing, and action noise. `leader_dt/rl/ppo_agent.py` builds Stable-Baselines3 PPO with rollout buffer, clipped policy updates, GAE, and value-function training. Both algorithms use the same environment interface.

### Policy wrappers and policy factory

`leader_dt/rl/wrappers.py` adapts Stable-Baselines3 models into the project policy interface. `leader_dt/evaluation/policy_factory.py` loads Greedy, TD3, and PPO policies into one dictionary for evaluation scripts.

The package `leader_dt/evaluation/__init__.py` resolves its exports lazily, so `leader_dt/rl/reward.py` can import `leader_dt.evaluation.penalized_objective` without pulling in the simulator and creating a circular import.

### Monte Carlo and sensitivity evaluation

`leader_dt/evaluation/monte_carlo.py` evaluates policies over many independent seeded scenarios under one fixed environment setting. `leader_dt/evaluation/sensitivity.py` repeats Monte Carlo evaluation across multiple values of one environment parameter.

### Plotting

`leader_dt/plotting/sensitivity_plots.py` should enforce the project thesis style by default. It uses readable labels, accessible colors, marker/line-style differentiation, tight bounding boxes, and PDF export for thesis inclusion.

## 4. Real-world analogy / visualization

Imagine a temporary traffic-control officer replacing a broken traffic light at a busy intersection. Every second, the officer must decide which lane gets to move next and how many cars are allowed through. The officer sees queues, urgency, road capacity, and congestion, but cannot serve everyone. Greedy always picks the lane that looks most urgent right now. TD3 and PPO are trained officers: they learn from repeated simulated days how to balance urgency, future congestion, and limited processing capacity. Monte Carlo is testing the officers over many different traffic days; sensitivity analysis changes one traffic condition at a time, such as heavier trucks, stricter timing, or more lanes, to see who remains reliable.
