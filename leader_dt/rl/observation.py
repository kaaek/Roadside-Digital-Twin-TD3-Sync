"""RL observation builder for exact sensor-vehicle pair scheduling."""
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
    - pair_count values: available data size normalized by max nominal payload
    - scalar CPU backlog ratio
    - scalar time progress t / T
    - scalar previous CPU load ratio
    - scalar urgency fraction
    """

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config

    def get_observation_dimension(self, scenario: Scenario) -> int:
        return 3 * scenario.pair_count + 4

    def build_observation_space(self, scenario: Scenario) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.get_observation_dimension(scenario),), dtype=np.float32)

    def build_observation(self, state: SimulationState, scenario: Scenario, feasible_pair_indices: list[int]) -> np.ndarray:
        system = self.simulation_config.system
        pair_count = scenario.pair_count
        aoi_normalized = np.clip(state.aoi_slots_array / max(system.freshness_threshold_slots, constants.EPSILON_FLOAT), 0.0, 1.0)
        feasible_mask = np.zeros(pair_count, dtype=np.float64)
        feasible_mask[np.asarray(feasible_pair_indices, dtype=int)] = 1.0
        slot_index = min(state.time_slot_index, system.time_horizon_slots - 1)
        data_sizes = scenario.available_data_size_bits_matrix[slot_index]
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
        urgency_fraction = float(np.mean(state.aoi_slots_array >= 0.7 * system.freshness_threshold_slots))
        return np.concatenate([
            aoi_normalized,
            feasible_mask,
            data_size_normalized,
            np.array([cpu_normalized, time_progress, previous_cpu_normalized, urgency_fraction], dtype=np.float64),
        ]).astype(np.float32)
