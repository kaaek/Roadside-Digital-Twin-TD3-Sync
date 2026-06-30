# TD3-RSU: Leader-Assisted Vehicular Digital Twin Synchronization

## Description

`TD3-RSU` is a Python research codebase for simulating a failed-RSU vehicular network where a leader vehicle collects sensor data from nearby vehicles to keep a Digital Twin fresh. The project compares a TD3 reinforcement learning policy against heuristic baselines using Monte Carlo evaluation, sensitivity sweeps, CPU-aware Greedy scheduling, and convergence diagnostics.

## Installation

1. Clone or enter the project directory:
    
    `cd ~/Data/Projects/RSU\ Research/src/td3-rsu`
    
2. Create a virtual environment:
    
    `python -m venv .venv`
    
3. Activate the virtual environment:
    
    `source .venv/bin/activate`
    
4. Upgrade packaging tools:
    
    `python -m pip install --upgrade pip setuptools wheel`
    
5. Install dependencies:
    
    `pip install -r requirements.txt`
    
6. Verify that the project compiles:
    
    `python -m compileall -q leader_dt scripts tests`
    

## Usage

### 1. Train TD3 with default configuration

Use this for a normal TD3 training run using the global configuration values in `leader_dt/constants.py` and `leader_dt/config.py`.

```bash
python scripts/train_td3.py
```

### 2. Train TD3 with convergence tracking

This command trains TD3, periodically evaluates the policy, saves the best model, saves the latest model, writes convergence metrics, and generates reward/AoI plots.

```bash
python scripts/train_td3_until_convergence.py --seed 1 --maximum-timesteps 3000000 --minimum-timesteps 3000000 --eval-frequency-steps 100000 --evaluation-episodes 50 --patience-evaluations 999999 --minimum-reward-improvement 50 --output-dir results/td3_tuned_3m_sigma005_seed1
```

### 3. Train TD3 with tuned hyperparameters

This command uses smaller action noise, explicit TD3 target smoothing, a larger replay buffer, a larger batch size, TensorBoard logging, Monitor logs, and periodic checkpoint saving.

```bash
python scripts/train_td3_until_convergence.py --seed 1 --maximum-timesteps 3000000 --minimum-timesteps 3000000 --eval-frequency-steps 100000 --evaluation-episodes 50 --patience-evaluations 999999 --minimum-reward-improvement 50 --action-noise-sigma 0.05 --target-policy-noise 0.20 --target-noise-clip 0.30 --buffer-size 1000000 --batch-size 256 --checkpoint-frequency-steps 250000 --output-dir results/td3_tuned_3m_sigma005_seed1 --tensorboard-log-dir results/td3_tuned_3m_sigma005_seed1/tensorboard --monitor-log-dir results/td3_tuned_3m_sigma005_seed1/monitor
```

### 4. Train TD3 with action noise `0.10`

Use this to compare a slightly larger exploration noise against the `0.05` setting.

```bash
python scripts/train_td3_until_convergence.py --seed 1 --maximum-timesteps 3000000 --minimum-timesteps 3000000 --eval-frequency-steps 100000 --evaluation-episodes 50 --patience-evaluations 999999 --minimum-reward-improvement 50 --action-noise-sigma 0.10 --target-policy-noise 0.20 --target-noise-clip 0.30 --buffer-size 1000000 --batch-size 256 --checkpoint-frequency-steps 250000 --output-dir results/td3_tuned_3m_sigma010_seed1 --tensorboard-log-dir results/td3_tuned_3m_sigma010_seed1/tensorboard --monitor-log-dir results/td3_tuned_3m_sigma010_seed1/monitor
```

### 5. Run a small smoke training test

Use this to quickly confirm that training, evaluation, logging, and checkpoint saving work.

```bash
python scripts/train_td3_until_convergence.py --seed 1 --maximum-timesteps 200000 --minimum-timesteps 100000 --eval-frequency-steps 50000 --evaluation-episodes 10 --patience-evaluations 5 --action-noise-sigma 0.05 --target-policy-noise 0.20 --target-noise-clip 0.30 --buffer-size 1000000 --batch-size 256 --checkpoint-frequency-steps 50000 --output-dir results/td3_tuned_smoke_sigma005_seed1 --tensorboard-log-dir results/td3_tuned_smoke_sigma005_seed1/tensorboard --monitor-log-dir results/td3_tuned_smoke_sigma005_seed1/monitor
```

### 6. Run Monte Carlo evaluation

This evaluates TD3 against the available baselines over multiple random trials.

```bash
python scripts/run_monte_carlo.py --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --trials 500 --seed-start 50000 --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

### 7. Run Monte Carlo evaluation with CPU-aware Greedy parameters

If the local version of `scripts/run_monte_carlo.py` supports CPU-aware Greedy CLI overrides, use:

```bash
python scripts/run_monte_carlo.py --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --trials 500 --seed-start 50000 --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

If the local script does not support these flags, use:

```bash
python scripts/run_monte_carlo.py --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --trials 500 --seed-start 50000 --output-dir results/td3_tuned_3m_sigma005_seed1/final_monte_carlo
```

### 8. Run all default sensitivity sweeps

```bash
python scripts/run_sensitivity.py --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --trials 100
```

### 9. Sensitivity sweep: defective zone size

This tests how policy performance changes as the RSU failure zone becomes smaller or larger.

```bash
python scripts/run_sensitivity.py --parameter zone_size_meter --values 200,400,800,1200,2000,3000 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 10. Sensitivity sweep: freshness threshold

This tests how performance changes when the AoI/freshness requirement becomes stricter or more relaxed.

```bash
python scripts/run_sensitivity.py --parameter freshness_threshold_slots --values 6,8,10,12,15,20 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 11. Sensitivity sweep: vehicle count

This tests how performance changes with fewer or more vehicles in the scenario.

```bash
python scripts/run_sensitivity.py --parameter vehicle_count --values 10,20,40,60,80 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 12. Sensitivity sweep: data-size / CPU pressure

This tests how TD3 and the baselines behave when sensor data sizes increase and CPU processing becomes more difficult.

```bash
python scripts/run_sensitivity.py --parameter data_size_high_multiplier --values 1.0,1.2,1.5,2.0,3.0,4.0 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 13. Sensitivity sweep: vehicle speed

This tests how mobility affects provider availability, communication, and Digital Twin freshness.

```bash
python scripts/run_sensitivity.py --parameter vehicle_speed_meter_per_second --values 5,10,15,20,30,40,50 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 14. Sensitivity sweep: sensors per vehicle

This tests how performance changes when each vehicle carries fewer or more sensor types.

```bash
python scripts/run_sensitivity.py --parameter sensors_per_vehicle --values 1,2,3,4 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 15. Sensitivity sweep: uplink bandwidth

This tests how communication capacity affects data collection, AoI, freshness violations, and CPU backlog.

```bash
python scripts/run_sensitivity.py --parameter uplink_bandwidth_hz --values 150000,300000,600000,900000,1200000,1800000 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 16. Sensitivity sweep: accuracy threshold

This tests how the minimum required upload accuracy affects policy behavior and constraint violations.

```bash
python scripts/run_sensitivity.py --parameter accuracy_threshold --values 0.6,0.7,0.8,0.9,1.0 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 17. Sensitivity sweep: time horizon

This tests how policy performance changes when the RSU failure duration is shorter or longer.

```bash
python scripts/run_sensitivity.py --parameter time_horizon_slots --values 20,40,60,80,100 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip
```

### 18. Sensitivity sweep with CPU-aware Greedy parameters

If the local version of `scripts/run_sensitivity.py` supports CPU-aware Greedy CLI overrides, use:

```bash
python scripts/run_sensitivity.py --parameter data_size_high_multiplier --values 1.0,1.5,2.0,3.0 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0
```

### 19. Minimum-accuracy CPU-aware Greedy sensitivity check

This tests whether CPU-aware Greedy becomes more CPU-efficient when it requests only the minimum required accuracy fraction.

```bash
python scripts/run_sensitivity.py --parameter data_size_high_multiplier --values 1.0,1.5,2.0,3.0 --trials 500 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 0.8
```

### 20. Run Z3 feasibility diagnostics

This checks feasibility of the scheduling and freshness constraints.

```bash
python scripts/check_z3_feasibility.py --vehicle-count 40 --sensors-per-vehicle 4 --time-horizon-slots 40 --freshness-threshold-slots 10 --timeout-ms 60000
```

### 21. View TensorBoard logs

```bash
tensorboard --logdir results/td3_tuned_3m_sigma005_seed1/tensorboard
```

### 22. Inspect generated output files

Training outputs are usually saved under:

`results/<run_name>/models/`

`results/<run_name>/metrics/`

`results/<run_name>/plots/`

`results/<run_name>/checkpoints/`

`results/<run_name>/tensorboard/`

`results/<run_name>/monitor/`

Monte Carlo outputs are usually saved under:

`results/<run_name>/final_monte_carlo/`

Sensitivity outputs are usually saved under:

`results/sensitivity/` or the `--output-dir` passed to `scripts/run_sensitivity.py`.

## License

MIT License

Copyright (c) 2026 Khalil El Kaaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.