"""Observation builder for exact sensor-vehicle pair scheduling."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import Scenario
from leader_dt.simulator.state import SimulationState

class ObservationBuilder:
    """Build normalized observations.

    Observation layout:
    - pair_count values: AoI_i(t) / tau_max
    - pair_count values: feasible-pair mask in Zone B
    - pair_count values: pending sample size normalized by max nominal payload
    - pair_count values: pending sample age / tau_max (only with a sample arrival process)
    - scalar CPU backlog ratio
    - scalar time progress t / T
    - scalar previous CPU load ratio
    - scalar urgency fraction
    """

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config

    def _pair_block_count(self) -> int:
        # The sample-age block is only added with an arrival process so legacy checkpoints keep their shape.
        return 4 if self.simulation_config.data_generation.has_sample_arrival_process else 3

    def get_observation_dimension(self, scenario: Scenario) -> int:
        return self._pair_block_count() * self.simulation_config.system.max_pair_count_for_action_space + 4

    def build_observation_space(self, scenario: Scenario) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.get_observation_dimension(scenario),), dtype=np.float32)

    def _pad_to_max_pair_count(self, value_array: np.ndarray) -> np.ndarray:
        max_pair_count = self.simulation_config.system.max_pair_count_for_action_space
        padded_array = np.zeros(max_pair_count, dtype=np.float64)
        input_array = np.asarray(value_array, dtype=np.float64).ravel()
        copy_count = min(max_pair_count, input_array.shape[0])
        padded_array[:copy_count] = input_array[:copy_count]
        return padded_array

    def build_observation(self, state: SimulationState, scenario: Scenario, feasible_pair_indices: list[int], active_pair_indices: list[int] | None = None) -> np.ndarray:
        system = self.simulation_config.system
        pair_count = scenario.pair_count
        active_pair_indices = active_pair_indices if active_pair_indices is not None else feasible_pair_indices
        active_mask = np.zeros(pair_count, dtype=bool)
        if len(active_pair_indices) > 0:
            active_pair_index_array = np.asarray(active_pair_indices, dtype=int)
            active_pair_index_array = active_pair_index_array[active_pair_index_array < pair_count]
            active_mask[active_pair_index_array] = True
        sensor_type_aoi_array = np.asarray(state.sensor_type_aoi_slots_array, dtype=np.float64)
        if sensor_type_aoi_array.shape[0] == scenario.sensor_type_count:
            pair_aoi_array = scenario.project_sensor_type_values_to_pairs(sensor_type_aoi_array)
        else:
            pair_aoi_array = state.aoi_slots_array
        aoi_normalized = np.clip(pair_aoi_array / max(system.freshness_threshold_slots, constants.EPSILON_FLOAT), 0.0, 1.0)
        aoi_normalized = np.where(active_mask, aoi_normalized, 0.0)
        feasible_mask = np.zeros(pair_count, dtype=np.float64)
        feasible_pair_index_array = np.asarray(feasible_pair_indices, dtype=int)
        feasible_pair_index_array = feasible_pair_index_array[feasible_pair_index_array < pair_count]
        if feasible_pair_index_array.size > 0:
            feasible_mask[feasible_pair_index_array] = 1.0
        data_sizes = np.asarray(state.pending_sample_size_bits_array, dtype=np.float64)
        data_sizes = np.where(active_mask, data_sizes, 0.0)
        max_nominal_data_size = max(sensor.nominal_data_size_bits for sensor in scenario.sensor_types)
        data_size_normalized = np.clip(data_sizes / max(max_nominal_data_size, constants.EPSILON_FLOAT), 0.0, 1.0)
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
        active_sensor_type_mask = np.zeros(scenario.sensor_type_count, dtype=bool)
        for pair in scenario.sensor_pair_index.pairs:
            if active_mask[int(pair.pair_id)]:
                active_sensor_type_mask[int(pair.sensor_type_id)] = True
        if np.any(active_sensor_type_mask) and sensor_type_aoi_array.shape[0] == scenario.sensor_type_count:
            urgency_fraction = float(np.mean(sensor_type_aoi_array[active_sensor_type_mask] >= 0.7 * system.freshness_threshold_slots))
        else:
            urgency_fraction = 0.0
        pair_blocks = [
            self._pad_to_max_pair_count(aoi_normalized),
            self._pad_to_max_pair_count(feasible_mask),
            self._pad_to_max_pair_count(data_size_normalized),
        ]
        if self._pair_block_count() == 4:
            generation_slot_array = np.asarray(state.pending_sample_generation_slot_array, dtype=np.float64)
            has_sample_mask = active_mask & ~np.isnan(generation_slot_array)
            sample_age_array = np.where(has_sample_mask, state.time_slot_index - generation_slot_array, 0.0)
            sample_age_normalized = np.clip(sample_age_array / max(system.freshness_threshold_slots, constants.EPSILON_FLOAT), 0.0, 1.0)
            pair_blocks.append(self._pad_to_max_pair_count(sample_age_normalized))
        return np.concatenate([
            *pair_blocks,
            np.array([cpu_normalized, time_progress, previous_cpu_normalized, urgency_fraction], dtype=np.float64),
        ]).astype(np.float32)
