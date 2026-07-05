# Skill: Train TD3

Refer to this document to train TD3, tune TD3 hyperparameters, run GPU/CPU benchmarks, generate convergence outputs, or explain TD3 training behavior.

## Relevant Files

- `leader_dt/rl/td3_agent.py`
- `leader_dt/rl/observation.py`
- `leader_dt/rl/reward.py`
- `leader_dt/rl/wrappers.py`
- `leader_dt/config.py`
- `leader_dt/constants.py`
- `scripts/train_td3.py`
- `scripts/train_td3_until_convergence.py`
- `leader_dt/plotting/convergence_plots.py`

## Explanation Format for Code Questions

1. One-sentence summary of what TD3 training code does.
2. High-level explanation of environment, actor, critics, replay buffer, action noise, and evaluation loop.
3. Block-by-block explanation of config, environment construction, model construction, training chunks, evaluation, checkpointing, and saved outputs.
4. Real-world analogy: TD3 is a trainee dispatcher learning from many simulated RSU-failure episodes which sensor provider to schedule and how much accuracy to request.

## Full TD3 Training Command

```bash
mkdir -p results/td3_zone_2km_3m_seed1_gpu

nohup python scripts/train_td3_until_convergence.py \
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
  --device cuda \
  --output-dir results/td3_zone_2km_3m_seed1_gpu \
  --tensorboard-log-dir results/td3_zone_2km_3m_seed1_gpu/tensorboard \
  --monitor-log-dir results/td3_zone_2km_3m_seed1_gpu/monitor \
  > results/td3_zone_2km_3m_seed1_gpu/training_stdout.log 2>&1 &
```

## Smoke Command

```bash
python scripts/train_td3_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 100000 \
  --minimum-timesteps 100000 \
  --eval-frequency-steps 50000 \
  --evaluation-episodes 5 \
  --patience-evaluations 999999 \
  --action-noise-sigma 0.05 \
  --target-policy-noise 0.20 \
  --target-noise-clip 0.30 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --checkpoint-frequency-steps 50000 \
  --device cuda \
  --output-dir results/td3_smoke_seed1_gpu \
  --tensorboard-log-dir results/td3_smoke_seed1_gpu/tensorboard \
  --monitor-log-dir results/td3_smoke_seed1_gpu/monitor
```

## Outputs

- `results/<run>/models/best_td3_exact_pair_zone_b.zip`
- `results/<run>/models/latest_td3_exact_pair_zone_b.zip`
- `results/<run>/metrics/td3_convergence_history.json`
- `results/<run>/metrics/td3_convergence_history.csv`
- `results/<run>/plots/td3_evaluation_reward_convergence.png`
- `results/<run>/plots/td3_evaluation_aoi_convergence.png`
- `results/<run>/td3_convergence_training_report.json`

## Interpretation Rules

- Use the best model for final Monte Carlo unless the user explicitly wants latest-model analysis.
- Do not claim convergence from raw reward only.
- Report smoothed reward plus AoI, freshness violations, CPU backlog, and accuracy violations.
- TD3 can use GPU if local benchmarking shows speedup.
