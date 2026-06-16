"""MILP variable indexing for b_i(t), x_i(t), A_i(t), and C(t)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MilpVariableIndex:
    pair_count: int
    time_horizon_slots: int

    def b_index(self, pair_index: int, time_slot_index: int) -> int:
        raise NotImplementedError

    def x_index(self, pair_index: int, time_slot_index: int) -> int:
        raise NotImplementedError

    def aoi_index(self, pair_index: int, time_slot_index: int) -> int:
        raise NotImplementedError

    def cpu_index(self, time_slot_index: int) -> int:
        raise NotImplementedError

    def total_variable_count(self) -> int:
        raise NotImplementedError