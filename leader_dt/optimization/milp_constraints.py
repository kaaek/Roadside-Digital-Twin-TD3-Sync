"""MILP constraints C1-C6 from the problem formulation."""

from __future__ import annotations

from dataclasses import dataclass

from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import Scenario
from leader_dt.optimization.milp_variables import MilpVariableIndex


class MilpConstraintBuilder:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        scenario: Scenario,
        variable_index: MilpVariableIndex,
    ) -> None:
        self.simulation_config = simulation_config
        self.scenario = scenario
        self.variable_index = variable_index

    def build_all_constraints(self):
        constraints = []
        constraints.extend(self.build_freshness_constraints_c1())
        constraints.extend(self.build_tdma_constraints_c2())
        constraints.extend(self.build_wireless_capacity_constraints_c3())
        constraints.extend(self.build_domain_constraints_c4())
        constraints.extend(self.build_terminal_cpu_constraint_c5())
        constraints.extend(self.build_data_availability_constraints_c6a())
        constraints.extend(self.build_accuracy_constraints_c6b())
        return constraints

    def build_freshness_constraints_c1(self):
        raise NotImplementedError

    def build_tdma_constraints_c2(self):
        raise NotImplementedError

    def build_wireless_capacity_constraints_c3(self):
        raise NotImplementedError

    def build_domain_constraints_c4(self):
        raise NotImplementedError

    def build_terminal_cpu_constraint_c5(self):
        raise NotImplementedError

    def build_data_availability_constraints_c6a(self):
        raise NotImplementedError

    def build_accuracy_constraints_c6b(self):
        raise NotImplementedError

    def build_aoi_transition_constraints(self):
        raise NotImplementedError

    def build_cpu_transition_constraints(self):
        raise NotImplementedError