"""Parameter sensitivity experiment interface."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SensitivityPointResult:
    parameter_name: str
    parameter_value: object
    policy_results: dict

class SensitivityEvaluator:
    def __init__(self, base_simulation_config) -> None:
        self.base_simulation_config = base_simulation_config

    def run_sweep(self, parameter_name: str, parameter_values: list, policy_dictionary: dict[str, object]) -> list[SensitivityPointResult]:
        raise NotImplementedError()
