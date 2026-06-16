"""Episode logging and rollout storage."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

@dataclass
class StepRecord:
    time_slot_index: int
    scheduled_pair_index: int | None
    collected_bits_float: float
    requested_accuracy_fraction_float: float
    achieved_accuracy_float: float
    weighted_aoi_float: float
    cpu_backlog_cycles_float: float
    freshness_violation_count: int
    accuracy_violation_count: int
    terminal_cpu_violation_count: int
    reward_float: float
    active_pair_count: int = 0

@dataclass
class EpisodeRecord:
    step_records: list[StepRecord] = field(default_factory=list)
    aoi_history_list: list[np.ndarray] = field(default_factory=list)
    active_pair_mask_history_list: list[np.ndarray] = field(default_factory=list)

    def append_step(self, step_record: StepRecord, aoi_slots_array: np.ndarray, active_pair_mask_array: np.ndarray | None = None) -> None:
        self.step_records.append(step_record)
        aoi_array = np.asarray(aoi_slots_array, dtype=np.float64).copy()
        self.aoi_history_list.append(aoi_array)
        if active_pair_mask_array is None:
            active_pair_mask_array = np.ones(aoi_array.shape[0], dtype=bool)
        self.active_pair_mask_history_list.append(np.asarray(active_pair_mask_array, dtype=bool).copy())

    def aoi_history_matrix(self) -> np.ndarray:
        if len(self.aoi_history_list) == 0:
            return np.empty((0, 0), dtype=np.float64)
        return np.vstack(self.aoi_history_list)

    def active_pair_mask_history_matrix(self) -> np.ndarray:
        if len(self.active_pair_mask_history_list) == 0:
            return np.empty((0, 0), dtype=bool)
        return np.vstack(self.active_pair_mask_history_list)

    def to_metric_dictionary(self) -> dict:
        if not self.step_records:
            return {}
        weighted_values = np.array([record.weighted_aoi_float for record in self.step_records], dtype=np.float64)
        aoi_history_matrix = self.aoi_history_matrix()
        active_mask_history_matrix = self.active_pair_mask_history_matrix()
        if aoi_history_matrix.size > 0 and active_mask_history_matrix.size > 0 and np.any(active_mask_history_matrix):
            maximum_aoi_float = float(np.max(aoi_history_matrix[active_mask_history_matrix]))
        else:
            maximum_aoi_float = 0.0
        accuracy_values = [record.achieved_accuracy_float for record in self.step_records if not np.isnan(record.achieved_accuracy_float)]
        return {
            "average_weighted_aoi_float": float(np.mean(weighted_values)),
            "maximum_aoi_float": maximum_aoi_float,
            "freshness_violation_count_integer": int(sum(record.freshness_violation_count for record in self.step_records)),
            "accuracy_violation_count_integer": int(sum(record.accuracy_violation_count for record in self.step_records)),
            "terminal_cpu_violation_count_integer": int(self.step_records[-1].terminal_cpu_violation_count),
            "final_cpu_backlog_cycles_float": float(self.step_records[-1].cpu_backlog_cycles_float),
            "total_collected_bits_float": float(sum(record.collected_bits_float for record in self.step_records)),
            "mean_accuracy_float": float(np.mean(accuracy_values)) if accuracy_values else float("nan"),
            "episode_return_float": float(sum(record.reward_float for record in self.step_records)),
        }
