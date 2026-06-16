"""Domain entities for vehicles and sensor types."""
from __future__ import annotations

from dataclasses import dataclass
from leader_dt.types import VehicleId, SensorTypeId

@dataclass(frozen=True)
class Vehicle:
    vehicle_id: VehicleId
    initial_position_meter: float
    speed_meter_per_second: float
    is_leader: bool = False

@dataclass(frozen=True)
class SensorType:
    sensor_type_id: SensorTypeId
    name: str
    priority_weight: float
    cpu_cycles_per_bit: float
    sensing_delay_slots: float
    nominal_data_size_bits: float
