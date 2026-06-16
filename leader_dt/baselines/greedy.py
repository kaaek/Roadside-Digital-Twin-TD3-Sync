"""Greedy exact-pair baselines."""
from __future__ import annotations

import numpy as np
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class GreedyWeightedAoiPolicy:
    """Select feasible pair with largest w_i A_i(t), request full data."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")
        feasible = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if not feasible:
            return action
        weights = environment.scenario.priority_weight_array_by_pair()
        scores = weights * environment.state.aoi_slots_array
        selected_pair = int(feasible[int(np.argmax(scores[feasible]))])
        action[selected_pair] = 1.0
        action[-1] = 1.0
        return action

class GreedyMaxAoiPolicy:
    """Select feasible pair with largest A_i(t), request full data."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")
        feasible = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if not feasible:
            return action
        selected_pair = int(feasible[int(np.argmax(environment.state.aoi_slots_array[feasible]))])
        action[selected_pair] = 1.0
        action[-1] = 1.0
        return action
