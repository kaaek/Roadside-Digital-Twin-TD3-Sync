"""MILP solver placeholder: MILP is excluded from the simulations for now because
it is computationally taxing. This module keeps a stable future interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MilpSolution:
    scheduled_pair_index_array: np.ndarray
    collected_bits_array: np.ndarray
    objective_value_float: float
    solver_status_string: str
    solve_time_seconds_float: float

class MilpScheduler:
    def solve(self, scenario):
        raise NotImplementedError("MILP is intentionally deferred for this phase of the project.")
