"""Monte Carlo evaluation over random seeds."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.evaluation.rollout import RolloutRunner

@dataclass(frozen=True)
class MonteCarloResult:
    policy_name: str
    metric_mean_dictionary: dict[str, float]
    metric_std_dictionary: dict[str, float]
    per_trial_metric_list: list[dict]

class MonteCarloEvaluator:
    def __init__(self, trial_count: int, seed_start: int = 1) -> None:
        self.trial_count = int(trial_count)
        self.seed_start = int(seed_start)
        self.rollout_runner = RolloutRunner()

    def evaluate_policy(self, policy_name: str, policy, environment_factory) -> MonteCarloResult:
        per_trial = []
        for trial_index in range(self.trial_count):
            seed = self.seed_start + trial_index
            env: LeaderSynchronizationEnv = environment_factory()
            metrics = self.rollout_runner.run_episode(env, policy, seed=seed)
            per_trial.append(metrics.__dict__)
        keys = per_trial[0].keys() if per_trial else []
        mean = {key: float(np.mean([row[key] for row in per_trial])) for key in keys}
        std = {key: float(np.std([row[key] for row in per_trial])) for key in keys}
        return MonteCarloResult(policy_name, mean, std, per_trial)

    def evaluate_policy_dictionary(self, policy_dictionary: dict[str, object], environment_factory) -> dict[str, MonteCarloResult]:
        return {name: self.evaluate_policy(name, policy, environment_factory) for name, policy in policy_dictionary.items()}
