"""Per-pair features scored by the feature-weight action and summarized in observations.

Every feature is computed the same way for every vehicle-sensor pair, so one
weight vector from the actor applies to all pairs regardless of how pair slots
are assigned in an episode.
"""
from __future__ import annotations

import numpy as np

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.simulator.state import SimulationState

PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "weighted_urgency",
    "reachable_accuracy",
    "proximity",
    "cpu_cost",
    "sample_age",
    "dwell",
)
PAIR_FEATURE_COUNT: int = len(PAIR_FEATURE_NAMES)

# Upper clip for the unbounded features (urgency, CPU cost); observations divide by it.
PAIR_FEATURE_CLIP_UPPER_BOUND: float = 3.0


class PairFeatureBuilder:
    """Build the ``pair_count × PAIR_FEATURE_COUNT`` feature matrix for one state.

    Columns, in ``PAIR_FEATURE_NAMES`` order:
    - weighted_urgency: w_s * A_s / (max_w * tau), clipped to [0, 3]
    - reachable_accuracy: min(1, uplink capacity / pending sample size); 0 without a sample
    - proximity: 1 - distance to leader / zone length
    - cpu_cost: cycles to process the collectable bits / per-slot CPU capacity, clipped to [0, 3]
    - sample_age: pending sample age / tau, clipped to [0, 1]
    - dwell: time until the carrier exits Zone B / horizon duration, clipped to [0, 1]
    """

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config

    def build(self, dynamics, state: SimulationState) -> np.ndarray:
        system = self.simulation_config.system
        road = self.simulation_config.road
        scenario = dynamics.scenario
        threshold = max(float(system.freshness_threshold_slots), constants.EPSILON_FLOAT)
        upper = PAIR_FEATURE_CLIP_UPPER_BOUND

        sensor_type_index_by_pair = scenario.sensor_type_index_array_by_pair()
        priority_weight_by_type = scenario.priority_weight_array_by_sensor_type()
        sensor_type_aoi = np.asarray(state.sensor_type_aoi_slots_array, dtype=np.float64)
        weighted_urgency = np.clip(
            priority_weight_by_type[sensor_type_index_by_pair]
            * sensor_type_aoi[sensor_type_index_by_pair]
            / (max(float(np.max(priority_weight_by_type)), constants.EPSILON_FLOAT) * threshold),
            0.0,
            upper,
        )

        pending_size_bits = dynamics.get_available_data_size_bits_array(state)
        uplink_capacity_bits = dynamics.compute_uplink_capacity_bits_array(state)
        has_sample = pending_size_bits > 0.0
        reachable_accuracy = np.where(
            has_sample,
            np.clip(uplink_capacity_bits / np.maximum(pending_size_bits, constants.EPSILON_FLOAT), 0.0, 1.0),
            0.0,
        )

        zone_length_meter = max(road.defective_zone_end_meter - road.defective_zone_start_meter, constants.EPSILON_FLOAT)
        proximity = np.clip(1.0 - dynamics.compute_distance_array_by_pair(state) / zone_length_meter, 0.0, 1.0)

        collectable_bits = np.minimum(pending_size_bits, uplink_capacity_bits)
        slot_cpu_capacity_cycles = max(system.leader_cpu_frequency_cycles_per_second * system.slot_duration_seconds, constants.EPSILON_FLOAT)
        cpu_cost = np.clip(collectable_bits * scenario.cpu_cycles_per_bit_array_by_pair() / slot_cpu_capacity_cycles, 0.0, upper)

        sample_age = np.clip(dynamics.get_pending_sample_age_slots_array(state) / threshold, 0.0, 1.0)

        vehicle_index_by_pair = scenario.vehicle_index_array_by_pair()
        position_meter = np.asarray(state.vehicle_positions_meter_array, dtype=np.float64)[vehicle_index_by_pair]
        speed_meter_per_second = scenario.vehicle_speed_meter_per_second_array[vehicle_index_by_pair]
        time_until_exit_seconds = np.maximum(road.defective_zone_end_meter - position_meter, 0.0) / np.maximum(speed_meter_per_second, constants.EPSILON_FLOAT)
        horizon_seconds = max(system.time_horizon_slots * system.slot_duration_seconds, constants.EPSILON_FLOAT)
        dwell = np.clip(time_until_exit_seconds / horizon_seconds, 0.0, 1.0)

        return np.column_stack([weighted_urgency, reachable_accuracy, proximity, cpu_cost, sample_age, dwell]).astype(np.float64)
