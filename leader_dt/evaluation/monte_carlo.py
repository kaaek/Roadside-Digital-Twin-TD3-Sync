"""Monte Carlo evaluation over random seeds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.evaluation.rollout import RolloutRunner

@dataclass(frozen=True)
class MonteCarloResult:
    policy_name: str
    metric_mean_dictionary: dict[str, float]
    metric_std_dictionary: dict[str, float]
    per_trial_metric_list: list[dict]
    # Std of the per-model means; only set when several trained models are pooled.
    metric_between_seed_std_dictionary: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class TrainedModelGroup:
    """Several trained models of one algorithm, evaluated and reported as one policy."""

    policies: list[Any]
    training_seeds: list[int] | None = None

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
        return {name: self.evaluate_policy_or_group(name, policy, environment_factory) for name, policy in policy_dictionary.items()}

    def evaluate_policy_or_group(self, policy_name: str, policy, environment_factory) -> MonteCarloResult:
        """Evaluate one policy, or every model of a ``TrainedModelGroup`` on the same seeds and pool them."""
        if not isinstance(policy, TrainedModelGroup):
            return self.evaluate_policy(policy_name, policy, environment_factory)
        model_results = [
            self.evaluate_policy(f"{policy_name} model {model_index}", model_policy, environment_factory)
            for model_index, model_policy in enumerate(policy.policies)
        ]
        return aggregate_monte_carlo_results(policy_name, model_results, policy.training_seeds)


def aggregate_monte_carlo_results(
    policy_name: str,
    result_list: list[MonteCarloResult],
    training_seeds: list[int] | None = None,
) -> MonteCarloResult:
    """Pool Monte Carlo results of several trained models into one result.

    Mean and std are taken over all pooled trials; the spread of the per-model
    means is stored in ``metric_between_seed_std_dictionary``.
    """
    if not result_list:
        raise ValueError("result_list must not be empty.")

    flattened_rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(result_list):
        training_seed = None if training_seeds is None else training_seeds[result_index]
        for row in result.per_trial_metric_list:
            enriched_row = dict(row)
            if training_seed is not None:
                enriched_row["training_seed_integer"] = int(training_seed)
            flattened_rows.append(enriched_row)

    metric_keys = [
        key
        for key in flattened_rows[0].keys()
        if key != "training_seed_integer" and isinstance(flattened_rows[0][key], (int, float, np.integer, np.floating))
    ]
    mean = {
        key: float(np.mean([float(row[key]) for row in flattened_rows]))
        for key in metric_keys
    }
    std = {
        key: float(np.std([float(row[key]) for row in flattened_rows]))
        for key in metric_keys
    }
    between_seed_std = {
        key: float(np.std([result.metric_mean_dictionary[key] for result in result_list]))
        for key in metric_keys
    }
    return MonteCarloResult(
        policy_name=policy_name,
        metric_mean_dictionary=mean,
        metric_std_dictionary=std,
        per_trial_metric_list=flattened_rows,
        metric_between_seed_std_dictionary=between_seed_std,
    )
