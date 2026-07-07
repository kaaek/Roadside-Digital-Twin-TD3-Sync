mkdir -p logs results/final

nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

TD3_MODEL="results/final/convergence_td3/models/best_td3.zip"
PPO_MODEL="results/final/convergence_ppo/models/best_ppo.zip"

echo "=== 1/5 Training default TD3 ==="
python scripts/train_td3_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 3000000 \
  --minimum-timesteps 3000000 \
  --eval-frequency-steps 100000 \
  --evaluation-episodes 50 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50.0 \
  --evaluation-seed-start 10000 \
  --checkpoint-frequency-steps 250000 \
  --action-noise-sigma 0.05 \
  --output-dir results/final/convergence_td3 \
  --tensorboard-log-dir results/final/convergence_td3/tensorboard \
  --monitor-log-dir results/final/convergence_td3/monitor \
  --checkpoint-output-dir results/final/convergence_td3/checkpoints

echo "=== 2/5 Training default PPO ==="
python scripts/train_ppo_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 3000000 \
  --minimum-timesteps 3000000 \
  --eval-frequency-steps 100000 \
  --evaluation-episodes 50 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50.0 \
  --evaluation-seed-start 10000 \
  --checkpoint-frequency-steps 250000 \
  --output-dir results/final/convergence_ppo \
  --tensorboard-log-dir results/final/convergence_ppo/tensorboard \
  --monitor-log-dir results/final/convergence_ppo/monitor \
  --checkpoint-output-dir results/final/convergence_ppo/checkpoints

echo "=== 3/5 Vehicle-count sensitivity sweep ==="
python scripts/run_sensitivity.py \
  --parameter vehicle_count \
  --values 10,20,40,60,80 \
  --trials 500 \
  --seed-start 50000 \
  --td3-model-path "$TD3_MODEL" \
  --ppo-model-path "$PPO_MODEL" \
  --output-dir results/final/sensitivity_vehicle_count

echo "=== 4/5 Data-size-high-multiplier sensitivity sweep ==="
python scripts/run_sensitivity.py \
  --parameter data_size_high_multiplier \
  --values 1.0,1.2,1.5,2.0,3.0,4.0 \
  --trials 500 \
  --seed-start 50000 \
  --td3-model-path "$TD3_MODEL" \
  --ppo-model-path "$PPO_MODEL" \
  --output-dir results/final/sensitivity_data_size_high_multiplier

echo "=== 5/5 Sensor-type scalability: train per point ==="
python scripts/run_sensor_type_scalability.py \
  --sensor-type-values 4,5,6,7,8,10,12,14,15,16 \
  --training-seeds 1,2,3 \
  --trials 500 \
  --seed-start 50000 \
  --maximum-timesteps 3000000 \
  --minimum-timesteps 3000000 \
  --eval-frequency-steps 100000 \
  --training-evaluation-episodes 50 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50.0 \
  --checkpoint-frequency-steps 250000 \
  --evaluation-seed-start 10000 \
  --td3-action-noise-sigma 0.05 \
  --output-dir results/final/sensor_type_scalability

echo "=== ALL FINAL RESULT COMMANDS COMPLETED ==="
' > logs/final_results.log 2>&1 &

echo $!