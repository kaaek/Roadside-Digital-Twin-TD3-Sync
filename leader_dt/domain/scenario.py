"""Scenario generation for defective RSU zone simulations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.domain.entities import Vehicle, SensorType
from leader_dt.domain.sensor_pairs import SensorPairIndex, SensorVehiclePair
from leader_dt.types import VehicleId, SensorTypeId, SensorPairId

@dataclass(frozen=True)
class Scenario:
    """
    All exogenous data for one episode. Vehicle positions move physically each slot.
    Available data sizes δ_i(t) are sampled per vehicle-sensor pair and per slot.
    """

    vehicles: list[Vehicle]
    sensor_types: list[SensorType]
    sensor_pair_index: SensorPairIndex
    initial_vehicle_positions_meter_array: np.ndarray
    vehicle_speed_meter_per_second_array: np.ndarray
    sensor_ownership_matrix: np.ndarray
    available_data_size_bits_matrix: np.ndarray  # shape: time_horizon × pair_count

    @property
    def pair_count(self) -> int:
        return self.sensor_pair_index.pair_count()

    def priority_weight_array_by_pair(self) -> np.ndarray:
        sensor_weight_by_type = {int(sensor.sensor_type_id): sensor.priority_weight for sensor in self.sensor_types}
        return np.array([
            sensor_weight_by_type[int(pair.sensor_type_id)]
            for pair in self.sensor_pair_index.pairs
        ], dtype=np.float64)

    def cpu_cycles_per_bit_array_by_pair(self) -> np.ndarray:
        cpu_by_type = {int(sensor.sensor_type_id): sensor.cpu_cycles_per_bit for sensor in self.sensor_types}
        return np.array([cpu_by_type[int(pair.sensor_type_id)] for pair in self.sensor_pair_index.pairs], dtype=np.float64)

    def sensing_delay_slots_array_by_pair(self) -> np.ndarray:
        delay_by_type = {int(sensor.sensor_type_id): sensor.sensing_delay_slots for sensor in self.sensor_types}
        return np.array([delay_by_type[int(pair.sensor_type_id)] for pair in self.sensor_pair_index.pairs], dtype=np.float64)

class ScenarioGenerator:
    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config
        self.random_generator = np.random.default_rng(simulation_config.random_seed)

    def generate(self, seed: int | None = None) -> Scenario:
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)
        vehicles = self._generate_vehicles()
        sensor_types = self._generate_sensor_types()
        ownership_matrix = self._generate_sensor_ownership_matrix(
            vehicle_count=len(vehicles),
            sensor_type_count=len(sensor_types),
        )
        sensor_pair_index = self._build_sensor_pair_index(ownership_matrix)
        data_sizes = self._generate_available_data_sizes(
            sensor_pair_index=sensor_pair_index,
            sensor_types=sensor_types,
            time_horizon_slots=self.simulation_config.system.time_horizon_slots,
        )
        return Scenario(
            vehicles=vehicles,
            sensor_types=sensor_types,
            sensor_pair_index=sensor_pair_index,
            initial_vehicle_positions_meter_array=np.array([v.initial_position_meter for v in vehicles], dtype=np.float64),
            vehicle_speed_meter_per_second_array=np.array([v.speed_meter_per_second for v in vehicles], dtype=np.float64),
            sensor_ownership_matrix=ownership_matrix,
            available_data_size_bits_matrix=data_sizes,
        )

    def _generate_vehicles(self) -> list[Vehicle]:
        system = self.simulation_config.system
        road = self.simulation_config.road
        zone_start = road.defective_zone_start_meter
        zone_end = road.defective_zone_end_meter
        positions = self.random_generator.uniform(zone_start, zone_end, size=system.vehicle_count)
        positions[0] = zone_start
        speeds = self.random_generator.normal(
            loc=road.vehicle_speed_meter_per_second,
            scale=road.vehicle_speed_jitter_std_meter_per_second,
            size=system.vehicle_count,
        )
        speeds = np.maximum(speeds, 0.1)
        # Make the leader traverse Zone B over the horizon by default.
        zone_length = max(zone_end - zone_start, 1.0)
        speeds[0] = zone_length / max(system.time_horizon_slots * system.slot_duration_seconds, constants.EPSILON_FLOAT)
        return [
            Vehicle(
                vehicle_id=VehicleId(index),
                initial_position_meter=float(positions[index]),
                speed_meter_per_second=float(speeds[index]),
                is_leader=(index == 0),
            )
            for index in range(system.vehicle_count)
        ]

    def _generate_sensor_types(self) -> list[SensorType]:
        sensor_definitions = constants.DEFAULT_SENSOR_DEFINITIONS[: self.simulation_config.system.sensor_type_count]
        return [
            SensorType(
                sensor_type_id=SensorTypeId(index),
                name=str(definition["name"]),
                priority_weight=float(definition["priority_weight"]),
                cpu_cycles_per_bit=float(definition["cpu_cycles_per_bit"]),
                sensing_delay_slots=float(definition["sensing_delay_slots"]),
                nominal_data_size_bits=float(definition["nominal_data_size_bits"]),
            )
            for index, definition in enumerate(sensor_definitions)
        ]

    def _generate_sensor_ownership_matrix(self, vehicle_count: int, sensor_type_count: int) -> np.ndarray:
        """Assign exactly `sensors_per_vehicle` sensor types to each vehicle.

        Keeping the number of owned sensors fixed per vehicle makes the number
        of sensor-vehicle pairs deterministic, which is important for Stable-
        Baselines3 because action/observation dimensions must not change across
        resets. The method retries until every sensor type appears at least once.
        """
        sensors_per_vehicle = min(self.simulation_config.system.sensors_per_vehicle, sensor_type_count)
        if vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive.")
        if sensors_per_vehicle <= 0:
            raise ValueError("sensors_per_vehicle must be positive.")

        max_attempt_count = 10_000
        for _ in range(max_attempt_count):
            ownership = np.zeros((vehicle_count, sensor_type_count), dtype=np.int32)
            for vehicle_index in range(vehicle_count):
                selected_sensor_indices = self.random_generator.choice(
                    np.arange(sensor_type_count),
                    size=sensors_per_vehicle,
                    replace=False,
                )
                ownership[vehicle_index, selected_sensor_indices] = 1
            if np.all(np.sum(ownership, axis=0) > 0):
                return ownership

        raise RuntimeError(
            "Failed to generate sensor ownership with global coverage. "
            "Increase vehicle_count or sensors_per_vehicle."
        )

    def _build_sensor_pair_index(self, sensor_ownership_matrix: np.ndarray) -> SensorPairIndex:
        pairs: list[SensorVehiclePair] = []
        pair_id = 0
        vehicle_count, sensor_type_count = sensor_ownership_matrix.shape
        for vehicle_index in range(vehicle_count):
            for sensor_type_index in range(sensor_type_count):
                if sensor_ownership_matrix[vehicle_index, sensor_type_index] == 1:
                    pairs.append(
                        SensorVehiclePair(
                            pair_id=SensorPairId(pair_id),
                            vehicle_id=VehicleId(vehicle_index),
                            sensor_type_id=SensorTypeId(sensor_type_index),
                        )
                    )
                    pair_id += 1
        return SensorPairIndex(pairs)

    def _generate_available_data_sizes(self, sensor_pair_index: SensorPairIndex, sensor_types: list[SensorType], time_horizon_slots: int) -> np.ndarray:
        data_config = self.simulation_config.data_generation
        nominal_by_type = {int(sensor.sensor_type_id): sensor.nominal_data_size_bits for sensor in sensor_types}
        data_size_matrix = np.zeros((time_horizon_slots, sensor_pair_index.pair_count()), dtype=np.float64)
        for pair in sensor_pair_index.pairs:
            nominal = nominal_by_type[int(pair.sensor_type_id)]
            data_size_matrix[:, int(pair.pair_id)] = self.random_generator.uniform(
                data_config.low_multiplier * nominal,
                data_config.high_multiplier * nominal,
                size=time_horizon_slots,
            )
        return data_size_matrix
