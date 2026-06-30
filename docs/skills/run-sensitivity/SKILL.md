# Skill: Run Sensitivity Sweeps

## Trigger

Use this skill when the user asks for robustness tests, parameter sweeps, environmental sensitivity, or commands for specific parameter sweeps.

## Relevant Files

- `scripts/run_sensitivity.py`
- `leader_dt/evaluation/sensitivity.py`
- `leader_dt/plotting/sensitivity_plots.py`
- `leader_dt/config.py`
- `leader_dt/constants.py`

## Command Template

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

## Recommended Sweeps

### Defective Zone Size

```bash
python scripts/run_sensitivity.py --parameter zone_size_meter --values 200,400,800,1200,2000,3000 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_zone_size
```

### Freshness Threshold

```bash
python scripts/run_sensitivity.py --parameter freshness_threshold_slots --values 6,8,10,12,15,20 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_freshness
```

### Vehicle Count

```bash
python scripts/run_sensitivity.py --parameter vehicle_count --values 10,20,40,60,80 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_vehicle_count
```

### Data Size / CPU Pressure

```bash
python scripts/run_sensitivity.py --parameter data_size_high_multiplier --values 1.0,1.2,1.5,2.0,3.0,4.0 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_data_pressure
```

### Vehicle Speed

```bash
python scripts/run_sensitivity.py --parameter vehicle_speed_meter_per_second --values 5,10,15,20,30,40,50 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_speed
```

### Sensors Per Vehicle

```bash
python scripts/run_sensitivity.py --parameter sensors_per_vehicle --values 1,2,3,4 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_sensors_per_vehicle
```

### Uplink Bandwidth

```bash
python scripts/run_sensitivity.py --parameter uplink_bandwidth_hz --values 150000,300000,600000,900000,1200000,1800000 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_bandwidth
```

### Accuracy Threshold

```bash
python scripts/run_sensitivity.py --parameter accuracy_threshold --values 0.6,0.7,0.8,0.9,1.0 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_accuracy
```

### Time Horizon

```bash
python scripts/run_sensitivity.py --parameter time_horizon_slots --values 20,40,60,80,100 --trials 500 --seed-start 50000 --model-path results/td3_tuned_3m_sigma005_seed1/models/best_td3_exact_pair_zone_b.zip --greedy-lambda-cpu 5 --greedy-requested-accuracy-fraction 1.0 --output-dir results/td3_tuned_3m_sigma005_seed1/sensitivity_horizon
```

## Rules

- Sweep one parameter at a time.
- Use the same model path and seed range across sensitivity points.
- Do not retrain TD3 inside sensitivity sweeps unless explicitly requested.
- Keep Greedy parameters fixed during environment sweeps.
- Use a separate `--output-dir` for each parameter.
