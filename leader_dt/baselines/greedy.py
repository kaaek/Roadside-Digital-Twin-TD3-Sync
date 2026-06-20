"""Greedy exact-pair baselines."""
from __future__ import annotations

import numpy as np
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class GreedyWeightedAoiPolicy:
    """Select feasible pair with largest w_s A_s(t), request full data."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")
        feasible = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if not feasible:
            return action
        sensor_type_weights = environment.scenario.priority_weight_array_by_sensor_type()
        sensor_type_aoi = environment.state.sensor_type_aoi_slots_array
        scores = np.zeros(environment.scenario.pair_count, dtype=np.float64)
        for pair in environment.scenario.sensor_pair_index.pairs:
            scores[int(pair.pair_id)] = sensor_type_weights[int(pair.sensor_type_id)] * sensor_type_aoi[int(pair.sensor_type_id)]
        selected_pair = int(feasible[int(np.argmax(scores[feasible]))])
        action[selected_pair] = 1.0
        action[-1] = 1.0
        return action

class GreedyMaxAoiPolicy:
    """Select feasible pair with largest A_s(t), request full data."""

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        if environment.state is None:
            raise RuntimeError("Environment must be reset before selecting an action.")
        feasible = environment.dynamics.get_feasible_pair_indices(environment.state)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if not feasible:
            return action
        sensor_type_aoi = environment.state.sensor_type_aoi_slots_array
        scores = np.zeros(environment.scenario.pair_count, dtype=np.float64)
        for pair in environment.scenario.sensor_pair_index.pairs:
            scores[int(pair.pair_id)] = sensor_type_aoi[int(pair.sensor_type_id)]
        selected_pair = int(feasible[int(np.argmax(scores[feasible]))])
        action[selected_pair] = 1.0
        action[-1] = 1.0
        return action
