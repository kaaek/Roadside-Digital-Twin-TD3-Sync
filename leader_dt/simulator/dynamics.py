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
        self.objective = WeightedAoiObjective(scenario.priority_weight_array_by_pair())
        self.topology = RoadTopology(
            lane_length_meter=simulation_config.road.lane_length_meter,
            defective_zone=DefectiveZone(simulation_config.road.defective_zone_start_meter, simulation_config.road.defective_zone_end_meter),
        )

    def initialize_state(self) -> SimulationState:
        return SimulationState(
            time_slot_index=0,
            vehicle_positions_meter_array=self.scenario.initial_vehicle_positions_meter_array.copy(),
            aoi_slots_array=np.ones(self.scenario.pair_count, dtype=np.float64),
            cpu_backlog_cycles_float=0.0,
            previous_cpu_added_cycles_float=0.0,
        )

    def advance_vehicle_positions(self, state: SimulationState) -> np.ndarray:
        return state.vehicle_positions_meter_array + self.scenario.vehicle_speed_meter_per_second_array * self.simulation_config.system.slot_duration_seconds

    def get_in_zone_vehicle_mask(self, state: SimulationState) -> np.ndarray:
        return self.topology.in_defective_zone_mask(state.vehicle_positions_meter_array)

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

    def step(self, state: SimulationState, action: SchedulingAction) -> tuple[SimulationState, dict]:
        slot_index = state.time_slot_index
        available_data_size_bits_array = self.scenario.available_data_size_bits_matrix[slot_index]
        uplink_capacity_bits_array = self.compute_uplink_capacity_bits_array(state)
        pair_index = action.scheduled_pair_index
        refresh_success = False
        achieved_accuracy = float("nan")
        transmission_delay_slots = 0.0
        sensing_delay_slots = 0.0
        added_cycles = 0.0
        if pair_index is not None and action.collected_bits_float > 0.0:
            pair_index = int(pair_index)
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
        next_aoi = self.aoi_transition_model.next_aoi_vector(
            current_aoi_slots_array=state.aoi_slots_array,
            scheduled_pair_index=pair_index,
            sensing_delay_slots_float=sensing_delay_slots,
            transmission_delay_slots_float=transmission_delay_slots,
            refresh_success_boolean=refresh_success,
        )
        next_cpu = self.cpu_backlog_model.next_backlog_cycles(
            state.cpu_backlog_cycles_float,
            added_cycles,
            self.simulation_config.system.slot_duration_seconds,
        )
        next_state = SimulationState(
            time_slot_index=state.time_slot_index + 1,
            vehicle_positions_meter_array=self.advance_vehicle_positions(state),
            aoi_slots_array=next_aoi,
            cpu_backlog_cycles_float=next_cpu,
            previous_cpu_added_cycles_float=added_cycles,
        )
        info = {
            "achieved_accuracy_float": achieved_accuracy,
            "refresh_success_boolean": refresh_success,
            "added_cycles_float": added_cycles,
            "weighted_aoi_float": self.objective.weighted_aoi_at_slot(next_aoi),
            "freshness_violation_count_integer": self.aoi_transition_model.count_freshness_violations(next_aoi),
            "accuracy_violation_count_integer": int(pair_index is not None and not refresh_success),
            "terminal_cpu_violation_count_integer": int(
                next_state.time_slot_index >= self.simulation_config.system.time_horizon_slots
                and next_state.cpu_backlog_cycles_float > 1.0e-9
            ),
        }
        return next_state, info
