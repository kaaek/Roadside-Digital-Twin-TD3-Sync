mkdir -p logs results/

nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

# One independently trained model per seed and algorithm; each seed gets its own
# output folder so best/latest .zip files are never overwritten.
# Override for a quick test: TRAINING_SEEDS=1 bash generate_final_results.sh
TRAINING_SEEDS="${TRAINING_SEEDS:-1 2 3 4 5}"
TD3_MODELS=""
PPO_MODELS=""

for SEED in $TRAINING_SEEDS; do
  TD3_DIR="results/convergence_td3/seed_${SEED}"
  PPO_DIR="results/convergence_ppo/seed_${SEED}"

  echo "=== Training TD3 seed ${SEED} ==="
  .venv/bin/python scripts/train_td3_until_convergence.py \
    --seed "$SEED" \
    --device cuda \
    --evaluation-seed-start 10000 \
    --output-dir "$TD3_DIR" \
    --tensorboard-log-dir "$TD3_DIR/tensorboard" \
    --monitor-log-dir "$TD3_DIR/monitor" \
    --checkpoint-output-dir "$TD3_DIR/checkpoints"

  echo "=== Training PPO seed ${SEED} ==="
  .venv/bin/python scripts/train_ppo_until_convergence.py \
    --seed "$SEED" \
    --device cpu \
    --evaluation-seed-start 10000 \
    --output-dir "$PPO_DIR" \
    --tensorboard-log-dir "$PPO_DIR/tensorboard" \
    --monitor-log-dir "$PPO_DIR/monitor" \
    --checkpoint-output-dir "$PPO_DIR/checkpoints"

  TD3_MODELS="${TD3_MODELS:+${TD3_MODELS},}${TD3_DIR}/models/best_td3.zip"
  PPO_MODELS="${PPO_MODELS:+${PPO_MODELS},}${PPO_DIR}/models/best_ppo.zip"
done

echo "TD3 models: $TD3_MODELS"
echo "PPO models: $PPO_MODELS"

echo "=== Vehicle-count sensitivity sweep ==="
.venv/bin/python scripts/run_sensitivity.py \
  --parameter vehicle_count \
  --values 10,20,40,60,80 \
  --trials 500 \
  --seed-start 50000 \
  --td3-model-path "$TD3_MODELS" \
  --ppo-model-path "$PPO_MODELS" \
  --output-dir results/sweeps/sensitivity_vehicle_count

echo "=== Data-size-high-multiplier sensitivity sweep ==="
.venv/bin/python scripts/run_sensitivity.py \
  --parameter data_size_high_multiplier \
  --values 1.0,1.2,1.5,2.0,3.0,4.0 \
  --trials 500 \
  --seed-start 50000 \
  --td3-model-path "$TD3_MODELS" \
  --ppo-model-path "$PPO_MODELS" \
  --output-dir results/sweeps/sensitivity_data_size_high_multiplier

# # echo "=== Sensor-type scalability: train per point ==="
# .venv/bin/python scripts/run_sensor_type_scalability.py \
#   --sensor-type-values 4,5,6,7,8,10,12,14,15,16 \
#   --training-seeds 1,2,3,4,5 \
#   --trials 500 \
#   --seed-start 50000 \
#   --maximum-timesteps 3000000 \
#   --minimum-timesteps 3000000 \
#   --eval-frequency-steps 100000 \
#   --training-evaluation-episodes 50 \
#   --patience-evaluations 999999 \
#   --minimum-reward-improvement 50.0 \
#   --checkpoint-frequency-steps 250000 \
#   --evaluation-seed-start 10000 \
#   --td3-action-noise-sigma 0.05 \
#   --output-dir results/sweeps/sensor_type_scalability

echo "=== ALL FINAL RESULT COMMANDS COMPLETED ==="
' > logs/final_results.log 2>&1 &

echo $!
