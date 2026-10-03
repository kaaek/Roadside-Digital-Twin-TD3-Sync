"""Observation builder for feature-weight pair scheduling."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import Scenario
from leader_dt.rl.pair_features import PAIR_FEATURE_CLIP_UPPER_BOUND, PAIR_FEATURE_COUNT, PAIR_FEATURE_NAMES
from leader_dt.simulator.state import SimulationState

# Per-sensor-type blocks are padded to the number of configured sensor definitions.
MAX_SENSOR_TYPE_COUNT_FOR_OBSERVATION: int = len(constants.DEFAULT_SENSOR_DEFINITIONS)
SENSOR_TYPE_BLOCK_COUNT: int = 4
GLOBAL_FEATURE_COUNT: int = 4
_UNBOUNDED_PAIR_FEATURE_MASK = np.array([name in ("weighted_urgency", "cpu_cost") for name in PAIR_FEATURE_NAMES])

class ObservationBuilder:
    """Build normalized observations whose size does not depend on the vehicle count.

    Observation layout:
    - per sensor type (padded to MAX_SENSOR_TYPE_COUNT_FOR_OBSERVATION):
      AoI / (3 tau) for active types, priority weight / max weight,
      has-a-feasible-provider flag, best reachable accuracy among feasible providers
    - mean and max of each pair feature over feasible pairs
    - scalar CPU backlog ratio
    - scalar time progress t / T
    - scalar previous CPU load ratio
    - scalar urgency fraction
    """

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config

    def get_observation_dimension(self, scenario: Scenario) -> int:
        return SENSOR_TYPE_BLOCK_COUNT * MAX_SENSOR_TYPE_COUNT_FOR_OBSERVATION + 2 * PAIR_FEATURE_COUNT + GLOBAL_FEATURE_COUNT

    def build_observation_space(self, scenario: Scenario) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.get_observation_dimension(scenario),), dtype=np.float32)

    def build_observation(
        self,
        state: SimulationState,
        scenario: Scenario,
        feasible_pair_indices: list[int],
        active_pair_indices: list[int] | None,
        pair_feature_matrix: np.ndarray,
    ) -> np.ndarray:
        system = self.simulation_config.system
        sensor_type_count = scenario.sensor_type_count
        sensor_type_index_by_pair = scenario.sensor_type_index_array_by_pair()
        active_pair_indices = active_pair_indices if active_pair_indices is not None else feasible_pair_indices
        active_sensor_type_mask = np.zeros(sensor_type_count, dtype=bool)
        active_sensor_type_mask[sensor_type_index_by_pair[np.asarray(active_pair_indices, dtype=int)]] = True

        sensor_type_aoi_array = np.asarray(state.sensor_type_aoi_slots_array, dtype=np.float64)
        threshold = max(float(system.freshness_threshold_slots), constants.EPSILON_FLOAT)
        aoi_normalized = np.where(active_sensor_type_mask, np.clip(sensor_type_aoi_array / (3.0 * threshold), 0.0, 1.0), 0.0)
        priority_weight_array = scenario.priority_weight_array_by_sensor_type()
        priority_weight_normalized = priority_weight_array / max(float(np.max(priority_weight_array)), constants.EPSILON_FLOAT)

        feasible_pair_index_array = np.asarray(feasible_pair_indices, dtype=int)
        has_feasible_provider = np.zeros(sensor_type_count, dtype=np.float64)
        best_reachable_accuracy = np.zeros(sensor_type_count, dtype=np.float64)
        feature_mean = np.zeros(PAIR_FEATURE_COUNT, dtype=np.float64)
        feature_max = np.zeros(PAIR_FEATURE_COUNT, dtype=np.float64)
        if feasible_pair_index_array.size > 0:
            feasible_features = np.asarray(pair_feature_matrix, dtype=np.float64)[feasible_pair_index_array]
            feasible_types = sensor_type_index_by_pair[feasible_pair_index_array]
            has_feasible_provider[feasible_types] = 1.0
            reachable_column = PAIR_FEATURE_NAMES.index("reachable_accuracy")
            np.maximum.at(best_reachable_accuracy, feasible_types, feasible_features[:, reachable_column])
            scale = np.where(_UNBOUNDED_PAIR_FEATURE_MASK, PAIR_FEATURE_CLIP_UPPER_BOUND, 1.0)
            feature_mean = np.mean(feasible_features, axis=0) / scale
            feature_max = np.max(feasible_features, axis=0) / scale

        def pad_to_max_sensor_types(value_array: np.ndarray) -> np.ndarray:
            padded_array = np.zeros(MAX_SENSOR_TYPE_COUNT_FOR_OBSERVATION, dtype=np.float64)
            padded_array[:sensor_type_count] = value_array
            return padded_array

        return np.clip(np.concatenate([
            pad_to_max_sensor_types(aoi_normalized),
            pad_to_max_sensor_types(priority_weight_normalized),
            pad_to_max_sensor_types(has_feasible_provider),
            pad_to_max_sensor_types(best_reachable_accuracy),
            feature_mean,
            feature_max,
            self._build_global_features(state, scenario, active_sensor_type_mask),
        ]), 0.0, 1.0).astype(np.float32)

    def _build_global_features(self, state: SimulationState, scenario: Scenario, active_sensor_type_mask: np.ndarray) -> np.ndarray:
        system = self.simulation_config.system
        sensor_type_aoi_array = np.asarray(state.sensor_type_aoi_slots_array, dtype=np.float64)
        cpu_normalized = np.clip(
            state.cpu_backlog_cycles_float
            / max(system.leader_cpu_frequency_cycles_per_second * system.time_horizon_slots * system.slot_duration_seconds, constants.EPSILON_FLOAT),
            0.0,
            1.0,
        )
        time_progress = np.clip(state.time_slot_index / max(system.time_horizon_slots, 1), 0.0, 1.0)
        previous_cpu_normalized = np.clip(
            state.previous_cpu_added_cycles_float
            / max(system.leader_cpu_frequency_cycles_per_second * system.slot_duration_seconds, constants.EPSILON_FLOAT),
            0.0,
            1.0,
        )
        if np.any(active_sensor_type_mask) and sensor_type_aoi_array.shape[0] == scenario.sensor_type_count:
            urgency_fraction = float(np.mean(sensor_type_aoi_array[active_sensor_type_mask] >= 0.7 * system.freshness_threshold_slots))
        else:
            urgency_fraction = 0.0
        return np.array([cpu_normalized, time_progress, previous_cpu_normalized, urgency_fraction], dtype=np.float64)
