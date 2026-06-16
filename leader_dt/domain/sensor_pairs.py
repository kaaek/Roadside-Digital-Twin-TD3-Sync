"""
Sensor-vehicle pair indexing; each sensor-vehicle pair (v, s) is a unique index i in set I.
This module provides that mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from leader_dt.types import VehicleId, SensorTypeId, SensorPairId

@dataclass(frozen=True)
class SensorVehiclePair:
    pair_id: SensorPairId
    vehicle_id: VehicleId
    sensor_type_id: SensorTypeId

class SensorPairIndex:
    """Bidirectional mapping between pair index i and (vehicle, sensor type)."""

    def __init__(self, pairs: list[SensorVehiclePair]) -> None:
        self.pairs = list(pairs)
        self._pair_by_id = {int(pair.pair_id): pair for pair in self.pairs}
        self._id_by_vehicle_sensor = {
            (int(pair.vehicle_id), int(pair.sensor_type_id)): int(pair.pair_id)
            for pair in self.pairs
        }

    def get_pair(self, pair_id: int) -> SensorVehiclePair:
        return self._pair_by_id[int(pair_id)]

    def get_pair_id(self, vehicle_id: int, sensor_type_id: int) -> int:
        return self._id_by_vehicle_sensor[(int(vehicle_id), int(sensor_type_id))]

    def pair_count(self) -> int:
        return len(self.pairs)

    def vehicle_ids(self) -> list[int]:
        return sorted({int(pair.vehicle_id) for pair in self.pairs})

    def sensor_type_ids(self) -> list[int]:
        return sorted({int(pair.sensor_type_id) for pair in self.pairs})

    def pairs_for_vehicle(self, vehicle_id: int) -> list[SensorVehiclePair]:
        return [pair for pair in self.pairs if int(pair.vehicle_id) == int(vehicle_id)]

    def pairs_for_sensor_type(self, sensor_type_id: int) -> list[SensorVehiclePair]:
        return [pair for pair in self.pairs if int(pair.sensor_type_id) == int(sensor_type_id)]
