"""Zone B (defective RSU) transition dynamics."""
from __future__ import annotations

import numpy as np

from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import Scenario
from leader_dt.domain.topology import DefectiveZone, RoadTopology
from leader_dt.models.communication import UplinkRateModel
from leader_dt.models.aoi import AoiTransitionModel
from leader_dt.models.cpu import CpuBacklogModel
from leader_dt.models.accuracy import AccuracyModel
from leader_dt.models.objective import WeightedAoiObjective
from leader_dt.simulator.action import SchedulingAction
from leader_dt.simulator.state import SimulationState

class LeaderSynchronizationDynamics:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        scenario: Scenario,
        uplink_rate_model: UplinkRateModel | None = None,
        aoi_transition_model: AoiTransitionModel | None = None,
        cpu_backlog_model: CpuBacklogModel | None = None,
        accuracy_model: AccuracyModel | None = None,
    ) -> None:
        self.simulation_config = simulation_config
        self.scenario = scenario
        self.uplink_rate_model = uplink_rate_model or UplinkRateModel(simulation_config.communication)
        self.aoi_transition_model = aoi_transition_model or AoiTransitionModel(simulation_config.system.freshness_threshold_slots)
        self.cpu_backlog_model = cpu_backlog_model or CpuBacklogModel(simulation_config.system.leader_cpu_frequency_cycles_per_second)
        self.accuracy_model = accuracy_model or AccuracyModel(simulation_config.system.accuracy_threshold)
        self.objective = WeightedAoiObjective(scenario.priority_weight_array_by_sensor_type())
        self.topology = RoadTopology(
            lane_length_meter=simulation_config.road.lane_length_meter,
            defective_zone=DefectiveZone(simulation_config.road.defective_zone_start_meter, simulation_config.road.defective_zone_end_meter),
        )

    def initialize_state(self) -> SimulationState:
        sensor_type_aoi_slots_array = np.ones(self.scenario.sensor_type_count, dtype=np.float64)
        return SimulationState(
            time_slot_index=0,
            vehicle_positions_meter_array=self.scenario.initial_vehicle_positions_meter_array.copy(),
            aoi_slots_array=self.scenario.project_sensor_type_values_to_pairs(sensor_type_aoi_slots_array),
            cpu_backlog_cycles_float=0.0,
            previous_cpu_added_cycles_float=0.0,
            cpu_backlog_by_pair_cycles_array=np.zeros(self.scenario.pair_count, dtype=np.float64),
            sensor_type_aoi_slots_array=sensor_type_aoi_slots_array,
        )

    def advance_vehicle_positions(self, state: SimulationState) -> np.ndarray:
        return state.vehicle_positions_meter_array + self.scenario.vehicle_speed_meter_per_second_array * self.simulation_config.system.slot_duration_seconds

    def get_in_zone_vehicle_mask(self, state: SimulationState) -> np.ndarray:
        return self.topology.in_defective_zone_mask(state.vehicle_positions_meter_array)

    def get_active_pair_mask(self, state: SimulationState) -> np.ndarray:
        in_zone_mask = self.get_in_zone_vehicle_mask(state)
        active_pair_mask = np.zeros(self.scenario.pair_count, dtype=bool)
        for pair in self.scenario.sensor_pair_index.pairs:
            active_pair_mask[int(pair.pair_id)] = bool(in_zone_mask[int(pair.vehicle_id)])
        return active_pair_mask

    def get_active_sensor_type_mask(self, state: SimulationState) -> np.ndarray:
        active_pair_mask = self.get_active_pair_mask(state)
        active_sensor_type_mask = np.zeros(self.scenario.sensor_type_count, dtype=bool)
        for pair in self.scenario.sensor_pair_index.pairs:
            if active_pair_mask[int(pair.pair_id)]:
                active_sensor_type_mask[int(pair.sensor_type_id)] = True
        return active_sensor_type_mask

    def get_active_pair_indices(self, state: SimulationState) -> list[int]:
        return np.where(self.get_active_pair_mask(state))[0].astype(int).tolist()

    def get_active_sensor_type_indices(self, state: SimulationState) -> list[int]:
        return np.where(self.get_active_sensor_type_mask(state))[0].astype(int).tolist()

    def get_feasible_pair_indices(self, state: SimulationState) -> list[int]:
        in_zone_mask = self.get_in_zone_vehicle_mask(state)
        feasible_pairs: list[int] = []
        for pair in self.scenario.sensor_pair_index.pairs:
            vehicle_id = int(pair.vehicle_id)
            if (not self.simulation_config.system.include_leader_as_provider) and vehicle_id == 0:
                continue
            if in_zone_mask[vehicle_id]:
                feasible_pairs.append(int(pair.pair_id))
        return feasible_pairs

    def compute_distance_array_by_pair(self, state: SimulationState) -> np.ndarray:
        leader_position = float(state.vehicle_positions_meter_array[0])
        distances = np.zeros(self.scenario.pair_count, dtype=np.float64)
        for pair in self.scenario.sensor_pair_index.pairs:
            pair_id = int(pair.pair_id)
            vehicle_position = float(state.vehicle_positions_meter_array[int(pair.vehicle_id)])
            distances[pair_id] = self.topology.distance_vehicle_to_leader(vehicle_position, leader_position, self.simulation_config.communication.reference_distance_meter)
        return distances

    def compute_uplink_rate_array_by_pair(self, state: SimulationState) -> np.ndarray:
        return self.uplink_rate_model.compute_rate_vector_bits_per_second(self.compute_distance_array_by_pair(state))

    def compute_uplink_capacity_bits_array(self, state: SimulationState) -> np.ndarray:
        return self.compute_uplink_rate_array_by_pair(state) * self.simulation_config.system.slot_duration_seconds

    def drain_cpu_backlog_by_pair(self, cpu_backlog_by_pair_cycles_array: np.ndarray, slot_duration_seconds: float) -> np.ndarray:
        backlog_array = np.asarray(cpu_backlog_by_pair_cycles_array, dtype=np.float64).copy()
        total_backlog_float = float(np.sum(backlog_array))
        processing_capacity_float = self.cpu_backlog_model.cpu_frequency_cycles_per_second * slot_duration_seconds
        if total_backlog_float <= processing_capacity_float:
            return np.zeros_like(backlog_array)
        remaining_ratio_float = (total_backlog_float - processing_capacity_float) / max(total_backlog_float, 1.0e-12)
        return backlog_array * remaining_ratio_float

    def step(self, state: SimulationState, action: SchedulingAction) -> tuple[SimulationState, dict]:
        slot_index = state.time_slot_index
        available_data_size_bits_array = self.scenario.available_data_size_bits_matrix[slot_index]
        pair_index = action.scheduled_pair_index
        scheduled_sensor_type_index: int | None = None
        refresh_success = False
        achieved_accuracy = float("nan")
        transmission_delay_slots = 0.0
        sensing_delay_slots = 0.0
        added_cycles = 0.0
        if pair_index is not None and action.collected_bits_float > 0.0:
            pair_index = int(pair_index)
            pair = self.scenario.sensor_pair_index.get_pair(pair_index)
            scheduled_sensor_type_index = int(pair.sensor_type_id)
            achieved_accuracy = self.accuracy_model.compute_accuracy(action.collected_bits_float, available_data_size_bits_array[pair_index])
            refresh_success = achieved_accuracy >= self.simulation_config.system.accuracy_threshold
            sensing_delay_slots = self.scenario.sensing_delay_slots_array_by_pair()[pair_index]
            transmission_delay_slots = self.aoi_transition_model.compute_transmission_delay_slots(
                action.collected_bits_float,
                self.compute_uplink_rate_array_by_pair(state)[pair_index],
                self.simulation_config.system.slot_duration_seconds,
            )
            added_cycles = self.cpu_backlog_model.compute_added_cycles(
                action.collected_bits_float,
                self.scenario.cpu_cycles_per_bit_array_by_pair()[pair_index],
            )
        current_sensor_type_aoi = np.asarray(state.sensor_type_aoi_slots_array, dtype=np.float64)
        if current_sensor_type_aoi.shape[0] != self.scenario.sensor_type_count:
            current_sensor_type_aoi = np.ones(self.scenario.sensor_type_count, dtype=np.float64)
        next_sensor_type_aoi = self.aoi_transition_model.next_sensor_type_aoi_vector(
            current_aoi_slots_array=current_sensor_type_aoi,
            scheduled_sensor_type_index=scheduled_sensor_type_index,
            sensing_delay_slots_float=sensing_delay_slots,
            transmission_delay_slots_float=transmission_delay_slots,
            refresh_success_boolean=refresh_success,
        )
        next_pair_aoi = self.scenario.project_sensor_type_values_to_pairs(next_sensor_type_aoi)
        current_cpu_by_pair = np.asarray(state.cpu_backlog_by_pair_cycles_array, dtype=np.float64)
        if current_cpu_by_pair.shape[0] != self.scenario.pair_count:
            current_cpu_by_pair = np.zeros(self.scenario.pair_count, dtype=np.float64)
        next_cpu_by_pair = current_cpu_by_pair.copy()
        if pair_index is not None and added_cycles > 0.0:
            next_cpu_by_pair[int(pair_index)] += added_cycles
        next_cpu_by_pair = self.drain_cpu_backlog_by_pair(next_cpu_by_pair, self.simulation_config.system.slot_duration_seconds)
        next_vehicle_positions = self.advance_vehicle_positions(state)
        preliminary_next_state = SimulationState(
            time_slot_index=state.time_slot_index + 1,
            vehicle_positions_meter_array=next_vehicle_positions,
            aoi_slots_array=next_pair_aoi,
            cpu_backlog_cycles_float=float(np.sum(next_cpu_by_pair)),
            previous_cpu_added_cycles_float=added_cycles,
            cpu_backlog_by_pair_cycles_array=next_cpu_by_pair,
            sensor_type_aoi_slots_array=next_sensor_type_aoi,
        )
        active_pair_mask = self.get_active_pair_mask(preliminary_next_state)
        active_sensor_type_mask = self.get_active_sensor_type_mask(preliminary_next_state)
        next_cpu_by_pair = next_cpu_by_pair.copy()
        next_cpu_by_pair[~active_pair_mask] = 0.0
        next_state = SimulationState(
            time_slot_index=preliminary_next_state.time_slot_index,
            vehicle_positions_meter_array=preliminary_next_state.vehicle_positions_meter_array,
            aoi_slots_array=preliminary_next_state.aoi_slots_array,
            cpu_backlog_cycles_float=float(np.sum(next_cpu_by_pair)),
            previous_cpu_added_cycles_float=added_cycles,
            cpu_backlog_by_pair_cycles_array=next_cpu_by_pair,
            sensor_type_aoi_slots_array=preliminary_next_state.sensor_type_aoi_slots_array,
        )
        info = {
            "achieved_accuracy_float": achieved_accuracy,
            "refresh_success_boolean": refresh_success,
            "added_cycles_float": added_cycles,
            "weighted_aoi_float": self.objective.weighted_aoi_at_slot(next_sensor_type_aoi, active_sensor_type_mask),
            "freshness_violation_count_integer": self.aoi_transition_model.count_freshness_violations(next_sensor_type_aoi, active_sensor_type_mask),
            "accuracy_violation_count_integer": int(pair_index is not None and not refresh_success),
            "terminal_cpu_violation_count_integer": int(
                next_state.time_slot_index >= self.simulation_config.system.time_horizon_slots
                and next_state.cpu_backlog_cycles_float > 1.0e-9
            ),
            "active_pair_count_integer": int(np.sum(active_pair_mask)),
            "active_sensor_type_count_integer": int(np.sum(active_sensor_type_mask)),
            "scheduled_sensor_type_index": scheduled_sensor_type_index,
        }
        return next_state, info
