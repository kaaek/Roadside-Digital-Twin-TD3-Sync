"""Random policy baseline."""
from __future__ import annotations

from leader_dt.simulator.action import PairSchedulingRequest
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class RandomPolicy:
    """Schedule a uniformly random feasible pair with a uniformly random accuracy request."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> PairSchedulingRequest:
        feasible_pair_indices = environment.dynamics.get_feasible_pair_indices(environment.state)
        requested_accuracy = float(environment.random_generator.uniform(0.0, 1.0))
        if not feasible_pair_indices:
            return PairSchedulingRequest(None, requested_accuracy)
        return PairSchedulingRequest(int(environment.random_generator.choice(feasible_pair_indices)), requested_accuracy)
