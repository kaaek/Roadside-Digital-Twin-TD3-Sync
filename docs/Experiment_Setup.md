# Experiment Setup, Constants, Config, and Current Results

## 1. Research aim summary

The experiment asks whether TD3 and PPO can learn robust continuous-control scheduling policies for leader-assisted Digital Twin updates during RSU failure, and whether they outperform or complement a CPU-aware Greedy heuristic under stochastic vehicular conditions. This is tested by training each RL model under a nominal scenario, evaluating them through Monte Carlo trials, and probing robustness through sensitivity sweeps.

## 2. High-level experiment approach and state flow

The environment stores the full simulator state: vehicle positions, active provider pairs, sensor-type AoI, each pair's pending sensor sample (size and generation slot), achieved accuracy, CPU backlog, time index, and violation counters. At every time slot, the policy consumes a normalized observation derived from this state and outputs an action. The action changes the simulator state by selecting a provider pair and requested accuracy; the environment then updates AoI, CPU backlog, collected bits, accuracy success, and violations.

Training modules consume the reward and next observation to update TD3 or PPO. Evaluation modules consume trained model checkpoints, wrap them as deterministic policies, and compare them against Greedy over repeated random seeds. Plotting modules consume saved metrics and produce thesis-style plots.

## 3. Block-by-block setup explanation

### Nominal simulator configuration

The nominal configuration defines the environment that TD3/PPO train on. It includes the defective zone length, episode horizon, vehicle count, number of sensor types, sensors per vehicle, freshness threshold, accuracy threshold, uplink bandwidth, CPU frequency, data-size distribution (0.5× to 2.0× the nominal payload), and sample arrival rate. Each pair generates Poisson($\lambda$) samples per slot and keeps only its latest unsent one; the nominal $\lambda = 0.10$ (`DEFAULT_SAMPLE_ARRIVAL_RATE_PER_SLOT`) gives each pair a new sample in about 10% of slots.

Physical values and their sources (all in `leader_dt/constants.py`):

| Quantity | Value | Basis |
|---|---|---|
| Pathloss | 32.4 + 20 log10(5.9) + 20 log10(d) dB | 3GPP TR 37.885, highway LOS |
| Shadowing | log-normal, σ = 3 dB, per vehicle and slot | 3GPP TR 37.885, highway LOS |
| Transmit power | 23 dBm | C-V2X UE power class |
| Noise | −174 dBm/Hz + 9 dB noise figure | thermal noise, 3GPP UE noise figure |
| Bandwidth | 1.8 MHz | one LTE-V2X sub-channel (10 RBs) of the 10 MHz ITS channel |
| Leader CPU | 2.0 GHz | one automotive SoC / OBU application core |
| Cycles per bit | 10–800 by sensor | MEC literature: hundreds for perception, tens for telemetry |
| Payloads | 0.25 kbit (fuel level) to 10 Mbit (LiDAR) | one compressed frame / point cloud, or one telemetry message |
| Sensing delay | 0.01–0.1 s | frame capture (~33 ms), LiDAR sweep / GNSS fix (~100 ms), bus read (~10 ms) |
| Vehicle speed | N(23, 4) m/s for every vehicle, leader included | highway traffic |

The default 8 sensor types include LiDAR so that heavy perception data is present. With these values the communication link and the leader CPU both constrain the schedule: on 40–50 seeds, CPU-aware Greedy leaves a CPU backlog at the end of about 22% of episodes and fails the accuracy threshold on about 5% of uploads (`scripts/diagnose_policy.py`).

### Randomness and seeds

Scenario randomness includes vehicle positions, sensor assignment, sample sizes, sample arrivals, and mobility conditions. Training uses a training seed. Evaluation uses independent seed ranges such as `50000, 50001, ...` so all policies face the same scenarios.

Seeding contract of `LeaderSynchronizationEnv.reset`: `reset(seed=s)` regenerates the scenario deterministically from `s`; `reset()` draws the next scenario from the same random stream. Stable-Baselines3 passes the training seed only on the first reset and auto-resets with no seed, so a training run with seed `k` sees a reproducible but different scenario every episode. (Before this contract, unseeded resets fell back to `SimulationConfig.random_seed`, so every training episode repeated one scenario; models trained before the fix are superseded.)

### TD3 training setup

TD3 is trained with an MLP actor/critic, Gaussian action noise, replay buffer, target policy smoothing, periodic evaluation, checkpoints, Monitor logs, and TensorBoard logs. The best model is selected by mean evaluation return over multiple evaluation episodes.

### PPO training setup

PPO is trained with an MLP actor-critic, on-policy rollouts, GAE, clipped policy updates, value-function loss, entropy coefficient, periodic evaluation, checkpoints, Monitor logs, and TensorBoard logs. PPO uses the same observation/action/reward formulation as TD3.

### Monte Carlo evaluation

Monte Carlo keeps the environment parameters fixed and runs many independent seeded scenarios. It produces means and standard deviations for metrics such as average weighted AoI, maximum AoI, freshness violations, accuracy violations, terminal CPU violations, final CPU backlog, total collected bits, mean accuracy, reward, and penalized score.

TD3 and PPO can be evaluated as a group of models, one per training seed: pass comma-separated checkpoint paths to `--td3-model-path` / `--ppo-model-path`. Every model runs on the same scenario seeds, the per-trial rows are pooled (tagged `training_seed_integer`, read from a `seed_<k>` path component), and `metric_between_seed_std_dictionary` holds the std of the per-model means. Sensitivity plots draw that between-seed std as a shaded band around the RL curves.

### Sensitivity sweeps

Sensitivity sweeps keep trained models fixed and change one environment parameter at a time. Each parameter value runs a full Monte Carlo evaluation. Current priority sweeps include vehicle count, task size (`data_size_high_multiplier`), sensors per vehicle, accuracy threshold, and sample arrival rate (`sample_arrival_rate`). One trained model can be evaluated across all of these sweeps, including vehicle count and arrival rate, because its 7-value action and 80-value observation do not depend on them.

### Final results pipeline

`generate_final_results.sh` runs the final pipeline in the background with the project interpreter (`.venv/bin/python`): for each training seed in `TRAINING_SEEDS` (default `1 2 3 4 5`) it trains TD3 (CUDA) and PPO (CPU) into `results/convergence_{td3,ppo}/seed_<k>/`, then runs the vehicle-count and task-size sweeps with all seeds' best models pooled per algorithm. The sensor-type scalability run is present but commented out. Output goes to `logs/final_results.log`; `check_progress.sh` follows it.

### Plotting standard

Sensitivity plots should use the thesis style by default: SciencePlots `science` and `grid` styles, serif fonts, colorblind-friendly colors, distinct markers and line styles, readable labels, tight bounding boxes, and both PNG/PDF export.