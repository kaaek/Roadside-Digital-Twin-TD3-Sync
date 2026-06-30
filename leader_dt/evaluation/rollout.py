"""Rollout execution for one policy and one scenario."""
from __future__ import annotations

from typing import Any

from leader_dt.evaluation.metrics import MetricCalculator, RolloutMetrics
from leader_dt.rl.wrappers import unwrap_leader_environment
from leader_dt.simulator.environment import LeaderSynchronizationEnv


class RolloutRunner:
    """Run complete episodes and compute project metrics."""

    def __init__(self, metric_calculator: MetricCalculator | None = None) -> None:
        """Create a runner with an optional custom metric calculator."""
        self.metric_calculator = metric_calculator or MetricCalculator()

    def run_episode(self, environment: LeaderSynchronizationEnv | Any, policy, seed: int | None = None) -> RolloutMetrics:
        """Run one episode using ``policy`` and return rollout metrics.

        ``environment`` may be either a raw ``LeaderSynchronizationEnv`` or a
        wrapper such as Stable-Baselines3 ``Monitor``.  Actions and metrics use
        the unwrapped leader environment, while ``reset`` and ``step`` are called
        on the provided wrapper so monitor logs are still recorded.
        """
        environment.reset(seed=seed)
        base_environment = unwrap_leader_environment(environment)
        done = False
        while not done:
            action = policy.select_action(base_environment)
            _, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
        return self.metric_calculator.compute_from_episode_record(base_environment.episode_record)
