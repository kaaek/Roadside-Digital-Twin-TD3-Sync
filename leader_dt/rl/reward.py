"""Reward function for TD3 without hard safety override."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.simulator.action import SchedulingAction
from leader_dt.simulator.state import SimulationState

@dataclass(frozen=True)
class RewardComponents:
    weighted_aoi_penalty: float
    violation_penalty: float
    proximity_penalty: float
    cpu_penalty: float
    accuracy_violation_penalty: float
    terminal_cpu_penalty: float
    accuracy_bonus: float

    @property
    def total_reward(self) -> float:
        return (
            -self.weighted_aoi_penalty
            -self.violation_penalty
            -self.proximity_penalty
            -self.cpu_penalty
            -self.accuracy_violation_penalty
            -self.terminal_cpu_penalty
            +self.accuracy_bonus
        )

class RewardCalculator:
    def __init__(self, simulation_config: SimulationConfig) -> None:
        self.simulation_config = simulation_config

    def compute_reward_components(
        self,
        state_after_action: SimulationState,
        action: SchedulingAction,
        achieved_accuracy_float: float,
        priority_weight_array: np.ndarray,
        weighted_aoi_float: float,
        freshness_violation_count_integer: int,
        accuracy_violation_count_integer: int,
        terminal_cpu_violation_count_integer: int,
    ) -> RewardComponents:
        system = self.simulation_config.system
        weights = np.asarray(priority_weight_array, dtype=np.float64)
        proximity_penalty = float(
            np.sum(weights * np.maximum(0.0, state_after_action.aoi_slots_array - 0.5 * system.freshness_threshold_slots) ** 2)
            / max(np.sum(weights) * system.freshness_threshold_slots, constants.EPSILON_FLOAT)
        )
        cpu_penalty = float(
            state_after_action.cpu_backlog_cycles_float
            / max(system.leader_cpu_frequency_cycles_per_second * system.slot_duration_seconds, constants.EPSILON_FLOAT)
        )
        accuracy_bonus = 0.0 if np.isnan(achieved_accuracy_float) else 2.0 * float(achieved_accuracy_float)
        return RewardComponents(
            weighted_aoi_penalty=float(weighted_aoi_float),
            violation_penalty=50.0 * float(freshness_violation_count_integer),
            proximity_penalty=proximity_penalty,
            cpu_penalty=cpu_penalty,
            accuracy_violation_penalty=20.0 * float(accuracy_violation_count_integer),
            terminal_cpu_penalty=100.0 * float(terminal_cpu_violation_count_integer),
            accuracy_bonus=accuracy_bonus,
        )

    def compute_reward(self, **kwargs) -> float:
        return self.compute_reward_components(**kwargs).total_reward
