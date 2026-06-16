"""Simulation state for one episode."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationState:
    time_slot_index: int
    vehicle_positions_meter_array: np.ndarray
    aoi_slots_array: np.ndarray
    cpu_backlog_cycles_float: float
    previous_cpu_added_cycles_float: float
    cpu_backlog_by_pair_cycles_array: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def copy(self) -> "SimulationState":
        return SimulationState(
            time_slot_index=self.time_slot_index,
            vehicle_positions_meter_array=self.vehicle_positions_meter_array.copy(),
            aoi_slots_array=self.aoi_slots_array.copy(),
            cpu_backlog_cycles_float=float(self.cpu_backlog_cycles_float),
            previous_cpu_added_cycles_float=float(self.previous_cpu_added_cycles_float),
            cpu_backlog_by_pair_cycles_array=np.asarray(self.cpu_backlog_by_pair_cycles_array, dtype=np.float64).copy(),
        )
