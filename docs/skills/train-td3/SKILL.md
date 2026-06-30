# Skill: Train TD3

## Trigger

Use this skill when the user asks to train TD3, tune TD3 hyperparameters, run a shorter/faster training experiment, reproduce a nuclear run, or generate convergence outputs.

## Relevant Files

- `leader_dt/rl/td3_agent.py`
- `leader_dt/rl/reward.py`
- `leader_dt/rl/observation.py`
- `leader_dt/config.py`
- `leader_dt/constants.py`
- `scripts/train_td3.py`
- `scripts/train_td3_until_convergence.py`
- `leader_dt/plotting/convergence_plots.py`

## Training Command Template

```bash
python scripts/train_td3_until_convergence.py \
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
  --output-dir results/td3_tuned_3m_sigma005_seed1 \
  --tensorboard-log-dir results/td3_tuned_3m_sigma005_seed1/tensorboard \
  --monitor-log-dir results/td3_tuned_3m_sigma005_seed1/monitor
```

## Quick Smoke Command

```bash
python scripts/train_td3_until_convergence.py \
  --seed 1 \
  --maximum-timesteps 200000 \
  --minimum-timesteps 100000 \
  --eval-frequency-steps 50000 \
  --evaluation-episodes 10 \
  --patience-evaluations 5 \
  --action-noise-sigma 0.05 \
  --target-policy-noise 0.20 \
  --target-noise-clip 0.30 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --checkpoint-frequency-steps 50000 \
  --output-dir results/td3_smoke_sigma005_seed1 \
  --tensorboard-log-dir results/td3_smoke_sigma005_seed1/tensorboard \
  --monitor-log-dir results/td3_smoke_sigma005_seed1/monitor
```

## Tuning Guidelines

- Prefer `--action-noise-sigma 0.05` and `0.10` for tuned tests.
- Prefer `--buffer-size 1000000` for serious runs.
- Prefer `--batch-size 256`; use `512` if runtime is acceptable.
- Keep `--target-policy-noise 0.20` and `--target-noise-clip 0.30` explicit.
- Use `--minimum-timesteps` equal to `--maximum-timesteps` when the user wants no early stopping.
- Use `--evaluation-episodes 50` for shorter serious runs and `100` for stronger convergence diagnostics.

## Outputs to Check

- `models/best_td3_exact_pair_zone_b.zip`
- `models/latest_td3_exact_pair_zone_b.zip`
- `metrics/td3_convergence_history.json`
- `metrics/td3_convergence_history.csv`
- `td3_convergence_training_report.json`
- `plots/td3_evaluation_reward_convergence.png`
- `plots/td3_evaluation_aoi_convergence.png`
- `checkpoints/td3_checkpoint_*_steps.zip`
- `tensorboard/`
- `monitor/`

## Validation

Run after code changes:

```bash
python -m compileall -q leader_dt scripts tests
pytest -q
```

## Common Pitfalls

- Do not claim convergence from a jagged raw reward plot.
- Do not compare TD3 against Greedy using different seed ranges.
- Do not forget to run Monte Carlo after training; training reward is not enough.
- If `best_td3_exact_pair_zone_b.zip` is missing, inspect the convergence script output directory and training log.
