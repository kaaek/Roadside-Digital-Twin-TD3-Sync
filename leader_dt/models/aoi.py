"""AoI transition model."""
from __future__ import annotations

import numpy as np
from leader_dt import constants

class AoiTransitionModel:
    def __init__(self, freshness_threshold_slots: int) -> None:
        self.freshness_threshold_slots = int(freshness_threshold_slots)

    def compute_transmission_delay_slots(self, collected_bits_float: float, uplink_rate_bits_per_second: float, slot_duration_seconds: float) -> float:
        denominator = max(uplink_rate_bits_per_second * slot_duration_seconds, constants.EPSILON_FLOAT)
        return float(collected_bits_float / denominator)

    def next_aoi_vector(self, current_aoi_slots_array: np.ndarray, scheduled_pair_index: int | None, sensing_delay_slots_float: float = 0.0, transmission_delay_slots_float: float = 0.0, refresh_success_boolean: bool = False) -> np.ndarray:
        next_aoi = np.asarray(current_aoi_slots_array, dtype=np.float64).copy() + 1.0
        if refresh_success_boolean and scheduled_pair_index is not None:
            next_aoi[int(scheduled_pair_index)] = float(sensing_delay_slots_float + transmission_delay_slots_float)
        return next_aoi

    def count_freshness_violations(self, aoi_slots_array: np.ndarray) -> int:
        return int(np.sum(np.asarray(aoi_slots_array) > self.freshness_threshold_slots))
