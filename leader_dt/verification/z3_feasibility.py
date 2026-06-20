"""Z3 feasibility checks for the paper's scheduling constraints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import importlib.util
import json
import time

import numpy as np

from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import Scenario, ScenarioGenerator
from leader_dt.domain.topology import DefectiveZone, RoadTopology
from leader_dt.models.communication import UplinkRateModel


@dataclass(frozen=True)
class StrictnessDiagnostics:
    time_horizon_slots: int
    pair_count: int
    sensor_type_count: int
    freshness_threshold_slots: int
    maximum_active_pair_count: int
    mean_active_pair_count: float
    maximum_active_sensor_type_count: int
    mean_active_sensor_type_count: float
    maximum_schedulable_pair_count: int
    slots_with_no_schedulable_pair_count: int
    maximum_persistent_pair_count_over_tau_window: int
    persistent_pair_capacity_violation_boolean: bool
    maximum_persistent_sensor_type_count_over_tau_window: int
    sensor_type_capacity_violation_boolean: bool
    total_minimum_cpu_cycles_if_one_refresh_per_slot: float
    total_cpu_capacity_cycles: float
    minimum_cpu_capacity_violation_boolean: bool

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Z3FeasibilityReport:
    z3_status_string: str
    elapsed_seconds_float: float
    seed_integer: int | None
    objective_value_float: float | None
    strictness_diagnostics_dictionary: dict[str, Any]
    relaxed_status_dictionary: dict[str, str]
    note_string: str

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dictionary(), indent=2, allow_nan=True))


@dataclass(frozen=True)
class ScenarioVerificationSnapshot:
    active_pair_mask_time_matrix: np.ndarray
    active_sensor_type_mask_time_matrix: np.ndarray
    uplink_capacity_bits_time_pair_matrix: np.ndarray
    available_data_size_bits_time_pair_matrix: np.ndarray
    priority_weight_by_sensor_type_array: np.ndarray
    cpu_cycles_per_bit_array: np.ndarray
    sensing_delay_slots_array: np.ndarray
    sensing_delay_slots_by_sensor_type_array: np.ndarray
    sensor_type_index_by_pair_array: np.ndarray


class Z3FeasibilityChecker:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        seed: int | None = None,
    ) -> None:
        if importlib.util.find_spec("z3") is None:
            raise ImportError(
                "The z3 Python package is not installed. Install it with: pip install z3-solver"
            )
        self.simulation_config = simulation_config
        self.seed = seed
        self.scenario = ScenarioGenerator(simulation_config).generate(seed=seed)
        self.snapshot = self._build_snapshot(self.scenario)

    def _build_snapshot(self, scenario: Scenario) -> ScenarioVerificationSnapshot:
        system_config = self.simulation_config.system
        road_config = self.simulation_config.road
        communication_config = self.simulation_config.communication
        horizon = system_config.time_horizon_slots
        pair_count = scenario.pair_count
        sensor_type_count = scenario.sensor_type_count
        topology = RoadTopology(
            lane_length_meter=road_config.lane_length_meter,
            defective_zone=DefectiveZone(
                start_meter=road_config.defective_zone_start_meter,
                end_meter=road_config.defective_zone_end_meter,
            ),
        )
        uplink_model = UplinkRateModel(communication_config)
        active_pair_mask_time_matrix = np.zeros((horizon + 1, pair_count), dtype=bool)
        active_sensor_type_mask_time_matrix = np.zeros((horizon + 1, sensor_type_count), dtype=bool)
        capacity_time_pair_matrix = np.zeros((horizon, pair_count), dtype=np.float64)
        sensor_type_index_by_pair_array = scenario.sensor_type_index_array_by_pair()

        for time_slot_index in range(horizon + 1):
            positions = (
                scenario.initial_vehicle_positions_meter_array
                + scenario.vehicle_speed_meter_per_second_array
                * system_config.slot_duration_seconds
                * time_slot_index
            )
            in_zone_vehicle_mask = topology.in_defective_zone_mask(positions)
            leader_position = float(positions[0])
            for pair in scenario.sensor_pair_index.pairs:
                pair_index = int(pair.pair_id)
                vehicle_index = int(pair.vehicle_id)
                sensor_type_index = int(pair.sensor_type_id)
                is_active = bool(in_zone_vehicle_mask[vehicle_index])
                active_pair_mask_time_matrix[time_slot_index, pair_index] = is_active
                if is_active:
                    active_sensor_type_mask_time_matrix[time_slot_index, sensor_type_index] = True
                if time_slot_index < horizon:
                    distance = topology.distance_vehicle_to_leader(
                        vehicle_position_meter=float(positions[vehicle_index]),
                        leader_position_meter=leader_position,
                        minimum_distance_meter=communication_config.reference_distance_meter,
                    )
                    capacity_time_pair_matrix[time_slot_index, pair_index] = (
                        uplink_model.compute_rate_bits_per_second(distance)
                        * system_config.slot_duration_seconds
                    )

        return ScenarioVerificationSnapshot(
            active_pair_mask_time_matrix=active_pair_mask_time_matrix,
            active_sensor_type_mask_time_matrix=active_sensor_type_mask_time_matrix,
            uplink_capacity_bits_time_pair_matrix=capacity_time_pair_matrix,
            available_data_size_bits_time_pair_matrix=scenario.available_data_size_bits_matrix,
            priority_weight_by_sensor_type_array=scenario.priority_weight_array_by_sensor_type(),
            cpu_cycles_per_bit_array=scenario.cpu_cycles_per_bit_array_by_pair(),
            sensing_delay_slots_array=scenario.sensing_delay_slots_array_by_pair(),
            sensing_delay_slots_by_sensor_type_array=scenario.sensing_delay_slots_array_by_sensor_type(),
            sensor_type_index_by_pair_array=sensor_type_index_by_pair_array,
        )

    def compute_strictness_diagnostics(self) -> StrictnessDiagnostics:
        system_config = self.simulation_config.system
        horizon = system_config.time_horizon_slots
        tau = system_config.freshness_threshold_slots
        eta = system_config.accuracy_threshold
        slot_duration = system_config.slot_duration_seconds
        active_pair_mask = self.snapshot.active_pair_mask_time_matrix[:horizon]
        active_sensor_type_mask = self.snapshot.active_sensor_type_mask_time_matrix[:horizon]
        active_pair_counts = np.sum(active_pair_mask, axis=1)
        active_sensor_type_counts = np.sum(active_sensor_type_mask, axis=1)
        schedulable_mask = (
            active_pair_mask
            & (
                eta * self.snapshot.available_data_size_bits_time_pair_matrix
                <= self.snapshot.uplink_capacity_bits_time_pair_matrix + 1.0e-9
            )
        )
        schedulable_counts = np.sum(schedulable_mask, axis=1)
        persistent_pair_count_list: list[int] = []
        persistent_sensor_type_count_list: list[int] = []
        window_length = min(tau + 1, horizon + 1)
        if window_length > 0:
            for start_index in range(0, horizon + 2 - window_length):
                pair_window_mask = self.snapshot.active_pair_mask_time_matrix[
                    start_index : start_index + window_length
                ]
                sensor_type_window_mask = self.snapshot.active_sensor_type_mask_time_matrix[
                    start_index : start_index + window_length
                ]
                persistent_pair_count_list.append(int(np.sum(np.all(pair_window_mask, axis=0))))
                persistent_sensor_type_count_list.append(int(np.sum(np.all(sensor_type_window_mask, axis=0))))
        maximum_persistent_pair_count = max(persistent_pair_count_list or [0])
        maximum_persistent_sensor_type_count = max(persistent_sensor_type_count_list or [0])

        minimum_added_cycle_list: list[float] = []
        for time_slot_index in range(horizon):
            candidate_indices = np.where(schedulable_mask[time_slot_index])[0]
            if candidate_indices.size == 0:
                continue
            minimum_cycles_this_slot = np.min(
                self.snapshot.cpu_cycles_per_bit_array[candidate_indices]
                * eta
                * self.snapshot.available_data_size_bits_time_pair_matrix[
                    time_slot_index, candidate_indices
                ]
            )
            minimum_added_cycle_list.append(float(minimum_cycles_this_slot))
        total_minimum_cpu_cycles = float(np.sum(minimum_added_cycle_list))
        total_cpu_capacity = float(
            system_config.leader_cpu_frequency_cycles_per_second
            * slot_duration
            * horizon
        )

        return StrictnessDiagnostics(
            time_horizon_slots=horizon,
            pair_count=self.scenario.pair_count,
            sensor_type_count=self.scenario.sensor_type_count,
            freshness_threshold_slots=tau,
            maximum_active_pair_count=int(np.max(active_pair_counts)) if active_pair_counts.size else 0,
            mean_active_pair_count=float(np.mean(active_pair_counts)) if active_pair_counts.size else 0.0,
            maximum_active_sensor_type_count=int(np.max(active_sensor_type_counts)) if active_sensor_type_counts.size else 0,
            mean_active_sensor_type_count=float(np.mean(active_sensor_type_counts)) if active_sensor_type_counts.size else 0.0,
            maximum_schedulable_pair_count=int(np.max(schedulable_counts)) if schedulable_counts.size else 0,
            slots_with_no_schedulable_pair_count=int(np.sum(schedulable_counts == 0)),
            maximum_persistent_pair_count_over_tau_window=maximum_persistent_pair_count,
            persistent_pair_capacity_violation_boolean=maximum_persistent_pair_count > tau,
            maximum_persistent_sensor_type_count_over_tau_window=maximum_persistent_sensor_type_count,
            sensor_type_capacity_violation_boolean=maximum_persistent_sensor_type_count > tau,
            total_minimum_cpu_cycles_if_one_refresh_per_slot=total_minimum_cpu_cycles,
            total_cpu_capacity_cycles=total_cpu_capacity,
            minimum_cpu_capacity_violation_boolean=total_minimum_cpu_cycles > total_cpu_capacity,
        )

    def check(
        self,
        timeout_milliseconds: int = 30_000,
        optimize_objective_boolean: bool = False,
        enforce_freshness_boolean: bool = True,
        enforce_terminal_cpu_boolean: bool = True,
    ) -> Z3FeasibilityReport:
        start_time = time.time()
        z3 = self._import_z3()
        strictness_diagnostics = self.compute_strictness_diagnostics()
        solver, variable_dictionary = self._build_z3_problem(
            z3=z3,
            timeout_milliseconds=timeout_milliseconds,
            optimize_objective_boolean=optimize_objective_boolean,
            enforce_freshness_boolean=enforce_freshness_boolean,
            enforce_terminal_cpu_boolean=enforce_terminal_cpu_boolean,
        )
        objective_value_float = None
        if optimize_objective_boolean:
            objective_expression = self._build_objective_expression(
                z3=z3,
                aoi_variables=variable_dictionary["A"],
            )
            solver.minimize(objective_expression)
        status = solver.check()
        if str(status) == "sat" and optimize_objective_boolean:
            model = solver.model()
            evaluated = model.eval(objective_expression, model_completion=True)
            objective_value_float = self._z3_number_to_float(evaluated)
        relaxed_status_dictionary = self._check_relaxed_statuses(timeout_milliseconds)
        note = self._build_note_string(str(status), strictness_diagnostics)
        return Z3FeasibilityReport(
            z3_status_string=str(status),
            elapsed_seconds_float=float(time.time() - start_time),
            seed_integer=self.seed,
            objective_value_float=objective_value_float,
            strictness_diagnostics_dictionary=strictness_diagnostics.to_dictionary(),
            relaxed_status_dictionary=relaxed_status_dictionary,
            note_string=note,
        )

    def _check_relaxed_statuses(self, timeout_milliseconds: int) -> dict[str, str]:
        status_dictionary: dict[str, str] = {}
        for label, freshness_boolean, terminal_cpu_boolean in (
            ("all_constraints", True, True),
            ("without_freshness_C1", False, True),
            ("without_terminal_cpu_C5", True, False),
            ("without_freshness_C1_and_terminal_cpu_C5", False, False),
        ):
            z3 = self._import_z3()
            solver, _ = self._build_z3_problem(
                z3=z3,
                timeout_milliseconds=timeout_milliseconds,
                optimize_objective_boolean=False,
                enforce_freshness_boolean=freshness_boolean,
                enforce_terminal_cpu_boolean=terminal_cpu_boolean,
            )
            status_dictionary[label] = str(solver.check())
        return status_dictionary

    def _build_z3_problem(
        self,
        z3,
        timeout_milliseconds: int,
        optimize_objective_boolean: bool,
        enforce_freshness_boolean: bool,
        enforce_terminal_cpu_boolean: bool,
    ):
        system_config = self.simulation_config.system
        horizon = system_config.time_horizon_slots
        pair_count = self.scenario.pair_count
        sensor_type_count = self.scenario.sensor_type_count
        tau = system_config.freshness_threshold_slots
        eta = system_config.accuracy_threshold
        slot_duration = system_config.slot_duration_seconds
        cpu_capacity_per_slot = (
            system_config.leader_cpu_frequency_cycles_per_second * slot_duration
        )
        z3_context = z3.Optimize() if optimize_objective_boolean else z3.Solver()
        z3_context.set(timeout=timeout_milliseconds)
        b = [[z3.Bool(f"b_{time_index}_{pair_index}") for pair_index in range(pair_count)] for time_index in range(horizon)]
        x = [[z3.Real(f"x_{time_index}_{pair_index}") for pair_index in range(pair_count)] for time_index in range(horizon)]
        a = [[z3.Real(f"A_{time_index}_{sensor_type_index}") for sensor_type_index in range(sensor_type_count)] for time_index in range(horizon + 1)]
        c = [z3.Real(f"C_{time_index}") for time_index in range(horizon + 1)]
        for sensor_type_index in range(sensor_type_count):
            z3_context.add(a[0][sensor_type_index] == self._real_value(z3, 1.0))
        z3_context.add(c[0] == self._real_value(z3, 0.0))
        for time_index in range(horizon):
            active_indices = np.where(self.snapshot.active_pair_mask_time_matrix[time_index])[0].astype(int).tolist()
            active_set = set(active_indices)
            if active_indices:
                z3_context.add(
                    z3.Sum([z3.If(b[time_index][pair_index], 1, 0) for pair_index in active_indices]) == 1
                )
            for pair_index in range(pair_count):
                if pair_index not in active_set:
                    z3_context.add(b[time_index][pair_index] == False)
                    z3_context.add(x[time_index][pair_index] == self._real_value(z3, 0.0))
                else:
                    z3_context.add(x[time_index][pair_index] >= self._real_value(z3, 0.0))
                    z3_context.add(
                        z3.Implies(
                            z3.Not(b[time_index][pair_index]),
                            x[time_index][pair_index] == self._real_value(z3, 0.0),
                        )
                    )
                    z3_context.add(
                        z3.Implies(
                            b[time_index][pair_index],
                            x[time_index][pair_index]
                            <= self._real_value(
                                z3,
                                self.snapshot.uplink_capacity_bits_time_pair_matrix[time_index, pair_index],
                            ),
                        )
                    )
                    z3_context.add(
                        z3.Implies(
                            b[time_index][pair_index],
                            x[time_index][pair_index]
                            <= self._real_value(
                                z3,
                                self.snapshot.available_data_size_bits_time_pair_matrix[time_index, pair_index],
                            ),
                        )
                    )
                    z3_context.add(
                        z3.Implies(
                            b[time_index][pair_index],
                            x[time_index][pair_index]
                            >= self._real_value(
                                z3,
                                eta * self.snapshot.available_data_size_bits_time_pair_matrix[time_index, pair_index],
                            ),
                        )
                    )
            added_cycles_expression = z3.Sum([
                self._real_value(z3, self.snapshot.cpu_cycles_per_bit_array[pair_index])
                * x[time_index][pair_index]
                for pair_index in range(pair_count)
            ])
            cpu_next_unclipped_expression = c[time_index] + added_cycles_expression - self._real_value(z3, cpu_capacity_per_slot)
            z3_context.add(
                c[time_index + 1]
                == z3.If(
                    cpu_next_unclipped_expression >= self._real_value(z3, 0.0),
                    cpu_next_unclipped_expression,
                    self._real_value(z3, 0.0),
                )
            )
            for sensor_type_index in range(sensor_type_count):
                candidate_pair_indices = [
                    pair_index for pair_index in active_indices
                    if int(self.snapshot.sensor_type_index_by_pair_array[pair_index]) == sensor_type_index
                ]
                reset_expression = a[time_index][sensor_type_index] + self._real_value(z3, 1.0)
                for pair_index in reversed(candidate_pair_indices):
                    capacity_bits = max(
                        float(self.snapshot.uplink_capacity_bits_time_pair_matrix[time_index, pair_index]),
                        1.0e-12,
                    )
                    reset_value = (
                        self._real_value(z3, self.snapshot.sensing_delay_slots_array[pair_index])
                        + x[time_index][pair_index] / self._real_value(z3, capacity_bits)
                    )
                    reset_expression = z3.If(b[time_index][pair_index], reset_value, reset_expression)
                z3_context.add(a[time_index + 1][sensor_type_index] == reset_expression)
        if enforce_freshness_boolean:
            for time_index in range(horizon + 1):
                for sensor_type_index in np.where(self.snapshot.active_sensor_type_mask_time_matrix[time_index])[0].astype(int).tolist():
                    z3_context.add(a[time_index][sensor_type_index] <= self._real_value(z3, tau))
        for time_index in range(horizon + 1):
            z3_context.add(c[time_index] >= self._real_value(z3, 0.0))
        if enforce_terminal_cpu_boolean:
            z3_context.add(c[horizon] == self._real_value(z3, 0.0))
        return z3_context, {"b": b, "x": x, "A": a, "C": c}

    def _build_objective_expression(self, z3, aoi_variables: list[list[Any]]) -> Any:
        horizon = self.simulation_config.system.time_horizon_slots
        weighted_terms = []
        normalizer_terms = []
        for time_index in range(1, horizon + 1):
            active_sensor_type_indices = np.where(self.snapshot.active_sensor_type_mask_time_matrix[time_index])[0].astype(int).tolist()
            for sensor_type_index in active_sensor_type_indices:
                weight = float(self.snapshot.priority_weight_by_sensor_type_array[sensor_type_index])
                weighted_terms.append(self._real_value(z3, weight) * aoi_variables[time_index][sensor_type_index])
                normalizer_terms.append(weight)
        if not weighted_terms:
            return self._real_value(z3, 0.0)
        denominator = max(float(np.sum(normalizer_terms)), 1.0e-12)
        return z3.Sum(weighted_terms) / self._real_value(z3, denominator)

    def _build_note_string(self, status_string: str, diagnostics: StrictnessDiagnostics) -> str:
        if status_string == "sat":
            return "The encoded hard constraints are jointly satisfiable for this generated scenario."
        if diagnostics.sensor_type_capacity_violation_boolean:
            return (
                "The constraints are likely too strict: more sensor types remain active across a freshness window "
                "than the TDMA scheduler can refresh one-at-a-time within tau slots."
            )
        if diagnostics.slots_with_no_schedulable_pair_count > 0:
            return (
                "The constraints are likely contradictory in at least one slot: no active pair can satisfy both the "
                "wireless-capacity and minimum-accuracy constraints."
            )
        if diagnostics.minimum_cpu_capacity_violation_boolean:
            return (
                "The CPU terminal constraint may be too strict: even the minimum possible one-refresh-per-slot workload "
                "exceeds total CPU capacity."
            )
        return "Z3 did not find a feasible assignment; inspect relaxed_status_dictionary for the first conflicting constraint group."

    @staticmethod
    def _import_z3():
        import z3

        return z3

    @staticmethod
    def _real_value(z3, value: float):
        return z3.RealVal(str(float(value)))

    @staticmethod
    def _z3_number_to_float(value: Any) -> float:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        if text.endswith("?"):
            text = text[:-1]
        return float(text)
