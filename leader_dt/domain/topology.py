"""Defective-zone topology and distance helpers."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DefectiveZone:
    start_meter: float
    end_meter: float

    @property
    def length_meter(self) -> float:
        return self.end_meter - self.start_meter

    def contains_position(self, position_meter: float) -> bool:
        return self.start_meter <= position_meter <= self.end_meter

class RoadTopology:
    def __init__(self, lane_length_meter: float, defective_zone: DefectiveZone) -> None:
        self.lane_length_meter = float(lane_length_meter)
        self.defective_zone = defective_zone

    def distance_vehicle_to_leader(self, vehicle_position_meter: float, leader_position_meter: float, minimum_distance_meter: float = 1.0) -> float:
        return float(max(abs(vehicle_position_meter - leader_position_meter), minimum_distance_meter))

    def in_defective_zone_mask(self, vehicle_positions_meter_array: np.ndarray) -> np.ndarray:
        return (
            (vehicle_positions_meter_array >= self.defective_zone.start_meter)
            & (vehicle_positions_meter_array <= self.defective_zone.end_meter)
        )
