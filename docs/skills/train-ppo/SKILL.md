# Skill: Train PPO

Refer to this document to train PPO, add PPO as an RL baseline, compare PPO to TD3, run PPO convergence analysis, or debug PPO training/evaluation.

## Relevant Files

- `leader_dt/rl/ppo_agent.py`
- `leader_dt/rl/observation.py`
- `leader_dt/rl/reward.py`
- `leader_dt/rl/wrappers.py`
- `leader_dt/config.py`
- `leader_dt/constants.py`
- `scripts/train_ppo.py`
- `scripts/train_ppo_until_convergence.py`
- `leader_dt/plotting/convergence_plots.py`

## Full PPO Training Command

```bash
mkdir -p results/ppo_zone_2km_3m_seed1_cpu

nohup python scripts/train_ppo_until_convergence.py \
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
  --monitor-log-dir results/ppo_zone_2km_3m_seed1_cpu/monitor \
  > results/ppo_zone_2km_3m_seed1_cpu/training_stdout.log 2>&1 &
```

## Smoke Command

```bash
python scripts/train_ppo_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 100000 \
  --minimum-timesteps 100000 \
  --eval-frequency-steps 50000 \
  --evaluation-episodes 5 \
  --patience-evaluations 999999 \
  --minimum-reward-improvement 50 \
  --learning-rate 0.0003 \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-range 0.20 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --checkpoint-frequency-steps 50000 \
  --device cpu \
  --output-dir results/ppo_smoke_seed1_cpu \
  --tensorboard-log-dir results/ppo_smoke_seed1_cpu/tensorboard \
  --monitor-log-dir results/ppo_smoke_seed1_cpu/monitor
```

## Outputs

- `results/<run>/models/best_ppo_exact_pair_zone_b.zip`
- `results/<run>/models/latest_ppo_exact_pair_zone_b.zip`
- `results/<run>/metrics/ppo_convergence_history.json`
- `results/<run>/metrics/ppo_convergence_history.csv`
- `results/<run>/plots/ppo_evaluation_reward_convergence.png`
- `results/<run>/plots/ppo_evaluation_aoi_convergence.png`
- `results/<run>/ppo_convergence_training_report.json`

## Notes

- PPO is on-policy and may require more timesteps than TD3 to become competitive.
- Stable-Baselines3 often recommends CPU for PPO with MLP policies; CUDA can be slower despite working.
- PPO should be evaluated with the same Monte Carlo seeds and metrics as TD3.
