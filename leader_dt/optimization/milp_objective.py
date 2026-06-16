"""MILP objective: normalized weighted AoI over all slots and pairs."""

from __future__ import annotations

import numpy as np

from leader_dt.optimization.milp_variables import MilpVariableIndex


class MilpObjectiveBuilder:
    def __init__(
        self,
        variable_index: MilpVariableIndex,
        priority_weight_array: np.ndarray,
    ) -> None:
        self.variable_index = variable_index
        self.priority_weight_array = priority_weight_array

    def build_objective_vector(self) -> np.ndarray:
        raise NotImplementedError