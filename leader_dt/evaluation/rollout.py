"""Rollout execution for one policy and one scenario."""
from __future__ import annotations

from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.evaluation.metrics import RolloutMetrics, MetricCalculator

class RolloutRunner:
    def __init__(self, metric_calculator: MetricCalculator | None = None) -> None:
        self.metric_calculator = metric_calculator or MetricCalculator()

    def run_episode(self, environment: LeaderSynchronizationEnv, policy, seed: int | None = None) -> RolloutMetrics:
        environment.reset(seed=seed)
        done = False
        while not done:
            action = policy.select_action(environment)
            _, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
        return self.metric_calculator.compute_from_episode_record(environment.episode_record)
