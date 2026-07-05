# Experiment Setup, Constants, Config, and Current Results

## 1. Research aim summary

The experiment asks whether TD3 and PPO can learn robust continuous-control scheduling policies for leader-assisted Digital Twin updates during RSU failure, and whether they outperform or complement a CPU-aware Greedy heuristic under stochastic vehicular conditions. This is tested by training each RL model under a nominal scenario, evaluating them through Monte Carlo trials, and probing robustness through sensitivity sweeps.

## 2. High-level experiment approach and state flow

The environment stores the full simulator state: vehicle positions, active provider pairs, sensor-type AoI, pair-level available data, achieved accuracy, CPU backlog, time index, and violation counters. At every time slot, the policy consumes a normalized observation derived from this state and outputs an action. The action changes the simulator state by selecting a provider pair and requested accuracy; the environment then updates AoI, CPU backlog, collected bits, accuracy success, and violations.

Training modules consume the reward and next observation to update TD3 or PPO. Evaluation modules consume trained model checkpoints, wrap them as deterministic policies, and compare them against Greedy over repeated random seeds. Plotting modules consume saved metrics and produce thesis-style plots.

## 3. Block-by-block setup explanation

### Nominal simulator configuration

The nominal configuration defines the environment that TD3/PPO train on. It includes the defective zone length, episode horizon, vehicle count, number of sensor types, sensors per vehicle, freshness threshold, accuracy threshold, uplink bandwidth, CPU frequency, and data-size distribution.

### Randomness and seeds

Scenario randomness includes vehicle positions, sensor assignment, available data sizes, and mobility conditions. Training uses a training seed. Evaluation uses independent seed ranges such as `50000, 50001, ...` so all policies face the same scenarios.

### TD3 training setup

TD3 is trained with an MLP actor/critic, Gaussian action noise, replay buffer, target policy smoothing, periodic evaluation, checkpoints, Monitor logs, and TensorBoard logs. The best model is selected by mean evaluation return over multiple evaluation episodes.

### PPO training setup

PPO is trained with an MLP actor-critic, on-policy rollouts, GAE, clipped policy updates, value-function loss, entropy coefficient, periodic evaluation, checkpoints, Monitor logs, and TensorBoard logs. PPO uses the same observation/action/reward formulation as TD3.

### Monte Carlo evaluation

Monte Carlo keeps the environment parameters fixed and runs many independent seeded scenarios. It produces means and standard deviations for metrics such as average weighted AoI, maximum AoI, freshness violations, accuracy violations, terminal CPU violations, final CPU backlog, total collected bits, mean accuracy, reward, and penalized score.

### Sensitivity sweeps

Sensitivity sweeps keep trained models fixed and change one environment parameter at a time. Each parameter value runs a full Monte Carlo evaluation. Current priority sweeps include task size (`data_size_high_multiplier`), sensors per vehicle, and accuracy threshold.

### Plotting standard

Sensitivity plots should use the thesis style by default: SciencePlots `science` and `grid` styles, serif fonts, colorblind-friendly colors, distinct markers and line styles, readable labels, tight bounding boxes, and both PNG/PDF export.