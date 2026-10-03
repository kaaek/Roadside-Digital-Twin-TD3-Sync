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


@dataclass(frozen=True)
class PairSchedulingRequest:
    """Heuristic decision: upload ``requested_accuracy_fraction`` of pair ``pair_index``'s
    pending sample, or stay idle when ``pair_index`` is None.

    Heuristic baselines return this instead of an RL action vector, so they behave
    the same under every RL action mode.
    """

    pair_index: int | None
    requested_accuracy_fraction: float


def build_scheduling_action(
    selected_pair: int | None,
    requested_accuracy: float,
    feasible_pair_indices: list[int],
    available_data_size_bits_array: np.ndarray,
    uplink_capacity_bits_array: np.ndarray,
) -> SchedulingAction:
    """Turn a pair choice and accuracy request into collected bits, or an idle slot."""
    requested_accuracy = float(np.clip(requested_accuracy, 0.0, 1.0))
    if selected_pair is None or requested_accuracy <= 0.0 or int(selected_pair) not in feasible_pair_indices:
        return SchedulingAction(None, 0.0, requested_accuracy, False)
    selected_pair = int(selected_pair)
    requested_bits = requested_accuracy * available_data_size_bits_array[selected_pair]
    collected_bits = min(requested_bits, uplink_capacity_bits_array[selected_pair], available_data_size_bits_array[selected_pair])
    return SchedulingAction(selected_pair, float(collected_bits), requested_accuracy, True)


class ActionDecoder:
    """Decode ``[feature weights..., requested accuracy]`` into one feasible pair.

    Each weight action in [0, 1] maps to w_k = 2a - 1 in [-1, 1].  Every feasible
    pair i is scored as sum_k w_k f_k(i) over the shared pair features, and the
    highest-scored pair is scheduled.  The action size does not depend on the
    number of vehicles or on how pair slots are assigned in an episode.
    """

    def __init__(self, feature_count: int) -> None:
        self.feature_count = int(feature_count)

    @property
    def action_dimension(self) -> int:
        return self.feature_count + 1

    def decode_feature_weights(self, raw_action_array: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action_array, dtype=np.float64).ravel()
        return 2.0 * np.clip(action[: self.feature_count], 0.0, 1.0) - 1.0

    def decode_rl_action(
        self,
        raw_action_array: np.ndarray,
        feasible_pair_indices: list[int],
        pair_feature_matrix: np.ndarray,
        available_data_size_bits_array: np.ndarray,
        uplink_capacity_bits_array: np.ndarray,
    ) -> SchedulingAction:
        action = np.asarray(raw_action_array, dtype=np.float64).ravel()
        if action.shape[0] != self.action_dimension:
            raise ValueError(f"Expected action dimension {self.action_dimension}, got {action.shape[0]}")
        requested_accuracy = float(np.clip(action[self.feature_count], 0.0, 1.0))
        if requested_accuracy <= 0.0 or len(feasible_pair_indices) == 0:
            return SchedulingAction(None, 0.0, requested_accuracy, False)
        feasible_pair_indices_array = np.asarray(feasible_pair_indices, dtype=int)
        scores = np.asarray(pair_feature_matrix, dtype=np.float64)[feasible_pair_indices_array] @ self.decode_feature_weights(action)
        selected_pair = int(feasible_pair_indices_array[np.argmax(scores)])
        return build_scheduling_action(
            selected_pair,
            requested_accuracy,
            feasible_pair_indices_array.tolist(),
            available_data_size_bits_array,
            uplink_capacity_bits_array,
        )
