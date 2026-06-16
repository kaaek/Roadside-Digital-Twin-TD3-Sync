"""Leader CPU backlog model."""
from __future__ import annotations

class CpuBacklogModel:
    def __init__(self, cpu_frequency_cycles_per_second: float) -> None:
        self.cpu_frequency_cycles_per_second = float(cpu_frequency_cycles_per_second)

    def compute_added_cycles(self, collected_bits_float: float, cpu_cycles_per_bit_float: float) -> float:
        return float(collected_bits_float * cpu_cycles_per_bit_float)

    def next_backlog_cycles(self, current_backlog_cycles_float: float, added_cycles_float: float, slot_duration_seconds: float) -> float:
        processed_cycles = self.cpu_frequency_cycles_per_second * slot_duration_seconds
        return max(0.0, current_backlog_cycles_float + added_cycles_float - processed_cycles)
