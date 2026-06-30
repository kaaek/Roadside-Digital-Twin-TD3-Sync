# Skill: TD3 Convergence Analysis

## Trigger

Use this skill when the user asks about TD3 convergence, erratic reward curves, nuclear-run JSON files, best checkpoints, reward smoothing, or professor-facing convergence explanations.

## Relevant Files

- `results/<run>/metrics/td3_convergence_history.json`
- `results/<run>/metrics/td3_convergence_history.csv`
- `results/<run>/td3_convergence_training_report.json`
- `results/<run>/plots/td3_evaluation_reward_convergence.png`
- `results/<run>/plots/td3_evaluation_aoi_convergence.png`
- `leader_dt/plotting/convergence_plots.py`
- `scripts/train_td3_until_convergence.py`

## Important Interpretation

A jagged reward curve does not automatically mean TD3 learned nothing, but it also does not prove convergence. In this project, reward can be noisy because the environment is stochastic and the reward contains large penalty terms for freshness, accuracy, and CPU violations.

Use these diagnostics instead of raw reward alone:

- Best-so-far reward.
- Moving-average reward.
- Moving-average AoI.
- Freshness violation trend.
- CPU backlog trend.
- Best checkpoint timestep.
- Final Monte Carlo comparison against baselines.

## JSON Summary Command

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in range(1, 6):
    path = Path(f"results/nuclear_td3/seed_{seed}/metrics/td3_convergence_history.json")
    if not path.exists():
        print(f"Missing {path}")
        continue

    data = json.loads(path.read_text())
    first = data[0]
    last = data[-1]
    best_reward = max(data, key=lambda x: x["mean_episode_return_float"])
    best_aoi = min(data, key=lambda x: x["average_weighted_aoi_float"])
    last_20 = data[-20:]

    print(f"\nSeed {seed}")
    print(f"  First reward: {first['mean_episode_return_float']:.2f} at {first['timesteps']:,}")
    print(f"  Last reward:  {last['mean_episode_return_float']:.2f} at {last['timesteps']:,}")
    print(f"  Best reward:  {best_reward['mean_episode_return_float']:.2f} at {best_reward['timesteps']:,}")
    print(f"  Best AoI:     {best_aoi['average_weighted_aoi_float']:.3f} at {best_aoi['timesteps']:,}")
    print(f"  Last-20 mean reward: {sum(x['mean_episode_return_float'] for x in last_20)/len(last_20):.2f}")
    print(f"  Last-20 mean AoI:    {sum(x['average_weighted_aoi_float'] for x in last_20)/len(last_20):.3f}")
PY
```

## Wording Guidance

Use this style:

```text
TD3 was trained extensively, but the reward curve does not show clean monotonic convergence or a perfectly flat plateau. The policy appears to fluctuate within a noisy performance band, which is expected in part because the environment is stochastic and the reward is penalty-heavy. Final Monte Carlo evaluation is therefore more reliable than visual inspection alone.
```

Avoid:

```text
TD3 clearly converged.
```

Avoid:

```text
The curve is noisy, therefore TD3 failed completely.
```

## Nuclear Run Context

Previous nuclear run:

- `5` independent seeds.
- `10,000,000` timesteps per seed.
- `50,000,000` total training timesteps.
- Evaluation every `100,000` timesteps.
- `100` evaluation episodes per checkpoint.
- Best checkpoint selected by highest mean evaluation return.
- Early stopping effectively disabled by setting minimum and maximum timesteps equal to `10,000,000` and using very large patience.
