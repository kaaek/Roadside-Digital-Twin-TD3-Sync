"""Weighted AoI objective."""
from __future__ import annotations

import numpy as np
from leader_dt import constants

class WeightedAoiObjective:
    def __init__(self, priority_weight_array: np.ndarray) -> None:
        self.priority_weight_array = np.asarray(priority_weight_array, dtype=np.float64)

    def weighted_aoi_at_slot(self, aoi_slots_array: np.ndarray) -> float:
        return float(np.sum(self.priority_weight_array * aoi_slots_array) / max(np.sum(self.priority_weight_array), constants.EPSILON_FLOAT))

    def average_weighted_aoi_over_horizon(self, aoi_time_pair_matrix: np.ndarray) -> float:
        matrix = np.asarray(aoi_time_pair_matrix, dtype=np.float64)
        if matrix.size == 0:
            return 0.0
        return float(np.mean([self.weighted_aoi_at_slot(row) for row in matrix]))
