"""Parameter sensitivity experiment interface."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from leader_dt.config import SimulationConfig
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator, MonteCarloResult

@dataclass(frozen=True)
class SensitivityPointResult:
    parameter_name: str
    parameter_value: object
    policy_results: dict

class SensitivityEvaluator:
    def __init__(self, base_simulation_config) -> None:
        self.base_simulation_config = base_simulation_config

    def run_sweep(self, parameter_name: str, parameter_values: list, policy_dictionary: dict[str, object]) -> list[SensitivityPointResult]:
        return self.run_sweep_with_trials(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            policy_dictionary=policy_dictionary,
            trial_count=1,
            seed_start=1,
        )

    def run_sweep_with_trials(
        self,
        parameter_name: str,
        parameter_values: list,
        policy_dictionary: dict[str, object],
        trial_count: int,
        seed_start: int = 1,
    ) -> list[SensitivityPointResult]:
        results: list[SensitivityPointResult] = []
        for parameter_value in parameter_values:
            simulation_config = self.build_config_with_parameter_value(parameter_name, parameter_value)
            environment_factory = self.make_environment_factory(simulation_config)
            monte_carlo_evaluator = MonteCarloEvaluator(trial_count=trial_count, seed_start=seed_start)
            policy_results = monte_carlo_evaluator.evaluate_policy_dictionary(policy_dictionary, environment_factory)
            results.append(
                SensitivityPointResult(
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    policy_results=policy_results,
                )
            )
        return results

    def make_environment_factory(self, simulation_config: SimulationConfig):
        from leader_dt.simulator.environment import LeaderSynchronizationEnv

        def environment_factory() -> LeaderSynchronizationEnv:
            return LeaderSynchronizationEnv(simulation_config)

        return environment_factory

    def build_config_with_parameter_value(self, parameter_name: str, parameter_value: Any) -> SimulationConfig:
        name = self._normalize_parameter_name(parameter_name)
        config = self.base_simulation_config
        if name.startswith("system."):
            field_name = name.split(".", 1)[1]
            return replace(config, system=replace(config.system, **{field_name: parameter_value}))
        if name.startswith("communication."):
            field_name = name.split(".", 1)[1]
            return replace(config, communication=replace(config.communication, **{field_name: parameter_value}))
        if name.startswith("road."):
            field_name = name.split(".", 1)[1]
            return replace(config, road=replace(config.road, **{field_name: parameter_value}))
        if name.startswith("data_generation."):
            field_name = name.split(".", 1)[1]
            return replace(config, data_generation=replace(config.data_generation, **{field_name: parameter_value}))
        if name == "zone_size_meter":
            center_meter = 0.5 * (config.road.defective_zone_start_meter + config.road.defective_zone_end_meter)
            half_size_meter = float(parameter_value) / 2.0
            return replace(
                config,
                road=replace(
                    config.road,
                    defective_zone_start_meter=center_meter - half_size_meter,
                    defective_zone_end_meter=center_meter + half_size_meter,
                ),
            )
        raise ValueError(f"Unsupported sensitivity parameter: {parameter_name}")

    def _normalize_parameter_name(self, parameter_name: str) -> str:
        alias_dictionary = {
            "vehicle_count": "system.vehicle_count",
            "sensor_type_count": "system.sensor_type_count",
            "sensors_per_vehicle": "system.sensors_per_vehicle",
            "time_horizon_slots": "system.time_horizon_slots",
            "freshness_threshold_slots": "system.freshness_threshold_slots",
            "accuracy_threshold": "system.accuracy_threshold",
            "uplink_bandwidth_hz": "communication.uplink_bandwidth_hz",
            "vehicle_speed_meter_per_second": "road.vehicle_speed_meter_per_second",
            "zone_size_meter": "zone_size_meter",
            "data_size_low_multiplier": "data_generation.low_multiplier",
            "data_size_high_multiplier": "data_generation.high_multiplier",
        }
        return alias_dictionary.get(parameter_name, parameter_name)
