"""Action representation and decoding."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SchedulingAction:
    scheduled_pair_index: int | None
    collected_bits_float: float
    requested_accuracy_fraction_float: float
    is_feasible_pair_boolean: bool

class ActionDecoder:
    """Decode TD3 continuous action into a choice of vehicle-sensor pair i
     to schedule and bits x_i(t).

    The actor emits one score per pair i in set I plus one accuracy fraction. The
    highest-scored feasible pair is selected by default. Accuracy is not hard
    clipped to eta_acc; violations are left to the reward/metrics.
    """

    def __init__(self, pair_count: int, max_pair_count: int | None = None) -> None:
        self.pair_count = int(pair_count)
        self.max_pair_count = int(max_pair_count if max_pair_count is not None else pair_count)
        if self.pair_count > self.max_pair_count:
            raise ValueError("pair_count cannot exceed max_pair_count.")

    @property
    def action_dimension(self) -> int:
        return self.max_pair_count + 1

    def decode_rl_action(
        self,
        raw_action_array: np.ndarray,
        feasible_pair_indices: list[int],
        available_data_size_bits_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
        deterministic: bool = True,
        random_generator: np.random.Generator | None = None,
    ) -> SchedulingAction:
        action = np.asarray(raw_action_array, dtype=np.float64).ravel()
        if action.shape[0] != self.max_pair_count + 1:
            raise ValueError(f"Expected action dimension {self.max_pair_count + 1}, got {action.shape[0]}")
        pair_scores = action[: self.pair_count]
        requested_accuracy = float(np.clip(action[self.max_pair_count], 0.0, 1.0))
        if requested_accuracy <= 0.0:
            return SchedulingAction(None, 0.0, requested_accuracy, False)
        if len(feasible_pair_indices) == 0:
            return SchedulingAction(None, 0.0, requested_accuracy, False)
        feasible_pair_indices_array = np.asarray(feasible_pair_indices, dtype=int)
        feasible_pair_indices_array = feasible_pair_indices_array[feasible_pair_indices_array < self.pair_count]
        if feasible_pair_indices_array.size == 0:
            return SchedulingAction(None, 0.0, requested_accuracy, False)
        if deterministic:
            selected_pair = int(feasible_pair_indices_array[np.argmax(pair_scores[feasible_pair_indices_array])])
        else:
            generator = random_generator or np.random.default_rng()
            clipped_scores = np.clip(pair_scores[feasible_pair_indices_array], 1e-8, None)
            probabilities = clipped_scores / np.sum(clipped_scores)
            selected_pair = int(generator.choice(feasible_pair_indices_array, p=probabilities))
        requested_bits = requested_accuracy * available_data_size_bits_array[selected_pair]
        collected_bits = min(requested_bits, uplink_capacity_bits_array[selected_pair], available_data_size_bits_array[selected_pair])
        return SchedulingAction(selected_pair, float(collected_bits), requested_accuracy, True)
