"""Smooth reward function for TD3/PPO without hard safety override."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.evaluation.penalized_objective import (
    DEFAULT_SMOOTH_REWARD_PENALTY_WEIGHTS,
    SmoothRewardPenaltyWeights,
    log1p_backlog_penalty,
    squared_positive_slack,
)
from leader_dt.simulator.action import SchedulingAction
from leader_dt.simulator.state import SimulationState


@dataclass(frozen=True)
class RewardComponents:
    """Additive reward components used only for training-time feedback."""

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
    """Compute smooth environment rewards while leaving reporting metrics unchanged."""

    def __init__(
        self,
        simulation_config: SimulationConfig,
        smooth_penalty_weights: SmoothRewardPenaltyWeights = DEFAULT_SMOOTH_REWARD_PENALTY_WEIGHTS,
    ) -> None:
        self.simulation_config = simulation_config
        self.smooth_penalty_weights = smooth_penalty_weights

    def _weighted_freshness_slack_penalty(
        self,
        state_after_action: SimulationState,
        priority_weight_array: np.ndarray,
    ) -> float:
        """Penalize how far sensor-type AoI exceeds the freshness threshold."""
        system = self.simulation_config.system
        weights = np.asarray(priority_weight_array, dtype=np.float64)
        aoi_array = np.asarray(
            state_after_action.sensor_type_aoi_slots_array,
            dtype=np.float64,
        )
        if aoi_array.shape[0] != weights.shape[0]:
            aoi_array = np.asarray(state_after_action.aoi_slots_array, dtype=np.float64)
        if aoi_array.shape[0] != weights.shape[0]:
            return 0.0

        threshold = max(float(system.freshness_threshold_slots), constants.EPSILON_FLOAT)
        normalized_excess_array = np.maximum(0.0, aoi_array - threshold) / threshold
        weighted_squared_excess = float(np.sum(weights * normalized_excess_array**2))
        return weighted_squared_excess / max(float(np.sum(weights)), constants.EPSILON_FLOAT)

    def _accuracy_shortfall_penalty(self, achieved_accuracy_float: float) -> float:
        """Penalize how far achieved accuracy falls below the configured target."""
        if np.isnan(achieved_accuracy_float):
            return 0.0
        target_accuracy = max(
            float(self.simulation_config.system.accuracy_threshold),
            constants.EPSILON_FLOAT,
        )
        normalized_shortfall = (target_accuracy - float(achieved_accuracy_float)) / target_accuracy
        return squared_positive_slack(normalized_shortfall)

    def _cpu_backlog_ratio(self, state_after_action: SimulationState) -> float:
        system = self.simulation_config.system
        slot_cpu_capacity = max(
            system.leader_cpu_frequency_cycles_per_second * system.slot_duration_seconds,
            constants.EPSILON_FLOAT,
        )
        return float(state_after_action.cpu_backlog_cycles_float / slot_cpu_capacity)

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
        del action, freshness_violation_count_integer, accuracy_violation_count_integer

        system = self.simulation_config.system
        penalty_weights = self.smooth_penalty_weights
        threshold = max(float(system.freshness_threshold_slots), constants.EPSILON_FLOAT)
        cpu_backlog_ratio = self._cpu_backlog_ratio(state_after_action)

        # Keep average weighted AoI in the reward, but normalize by the freshness
        # threshold so it does not dominate every other smooth component.
        weighted_aoi_penalty = float(weighted_aoi_float) / threshold

        freshness_slack_penalty = self._weighted_freshness_slack_penalty(
            state_after_action=state_after_action,
            priority_weight_array=priority_weight_array,
        )

        # Keep a softer version of the old proximity/urgency shaping term.  This
        # starts before the threshold but is normalized and capped by the squared
        # excess shape instead of a hard violation jump.
        weights = np.asarray(priority_weight_array, dtype=np.float64)
        aoi_array = np.asarray(state_after_action.sensor_type_aoi_slots_array, dtype=np.float64)
        if aoi_array.shape[0] != weights.shape[0]:
            aoi_array = np.asarray(state_after_action.aoi_slots_array, dtype=np.float64)
        if aoi_array.shape[0] == weights.shape[0]:
            urgency_slack = np.maximum(0.0, aoi_array - 0.5 * threshold) / threshold
            proximity_penalty = float(
                np.sum(weights * urgency_slack**2)
                / max(float(np.sum(weights)), constants.EPSILON_FLOAT)
            )
        else:
            proximity_penalty = 0.0

        cpu_penalty = penalty_weights.cpu_backlog_weight * log1p_backlog_penalty(
            cpu_backlog_ratio,
        )
        accuracy_penalty = (
            penalty_weights.accuracy_shortfall_weight
            * self._accuracy_shortfall_penalty(achieved_accuracy_float)
        )
        terminal_cpu_penalty = 0.0
        if terminal_cpu_violation_count_integer:
            terminal_cpu_penalty = (
                penalty_weights.terminal_cpu_backlog_weight
                * log1p_backlog_penalty(cpu_backlog_ratio)
            )
        accuracy_bonus = 0.0 if np.isnan(achieved_accuracy_float) else 2.0 * float(achieved_accuracy_float)

        return RewardComponents(
            weighted_aoi_penalty=weighted_aoi_penalty,
            violation_penalty=penalty_weights.freshness_slack_weight * freshness_slack_penalty,
            proximity_penalty=proximity_penalty,
            cpu_penalty=cpu_penalty,
            accuracy_violation_penalty=accuracy_penalty,
            terminal_cpu_penalty=terminal_cpu_penalty,
            accuracy_bonus=accuracy_bonus,
        )

    def compute_reward(self, **kwargs) -> float:
        return self.compute_reward_components(**kwargs).total_reward
