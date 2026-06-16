"""Weighted AoI objective."""
from __future__ import annotations

import numpy as np
from leader_dt import constants

class WeightedAoiObjective:
    def __init__(self, priority_weight_array: np.ndarray) -> None:
        self.priority_weight_array = np.asarray(priority_weight_array, dtype=np.float64)

    def weighted_aoi_at_slot(self, aoi_slots_array: np.ndarray, active_pair_mask_array: np.ndarray | None = None) -> float:
        aoi_array = np.asarray(aoi_slots_array, dtype=np.float64)
        weight_array = self.priority_weight_array
        if active_pair_mask_array is not None:
            active_mask = np.asarray(active_pair_mask_array, dtype=bool)
            if active_mask.shape[0] != aoi_array.shape[0]:
                raise ValueError("active_pair_mask_array must have the same length as aoi_slots_array.")
            if not np.any(active_mask):
                return 0.0
            aoi_array = aoi_array[active_mask]
            weight_array = weight_array[active_mask]
        return float(np.sum(weight_array * aoi_array) / max(np.sum(weight_array), constants.EPSILON_FLOAT))

    def average_weighted_aoi_over_horizon(self, aoi_time_pair_matrix: np.ndarray, active_pair_mask_time_matrix: np.ndarray | None = None) -> float:
        matrix = np.asarray(aoi_time_pair_matrix, dtype=np.float64)
        if matrix.size == 0:
            return 0.0
        if active_pair_mask_time_matrix is None:
            return float(np.mean([self.weighted_aoi_at_slot(row) for row in matrix]))
        mask_matrix = np.asarray(active_pair_mask_time_matrix, dtype=bool)
        return float(np.mean([
            self.weighted_aoi_at_slot(matrix[row_index], mask_matrix[row_index])
            for row_index in range(matrix.shape[0])
        ]))
