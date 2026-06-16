"""Gymnasium environment wrapper around the defective RSU zone simulator."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from leader_dt.config import SimulationConfig
from leader_dt.domain.scenario import ScenarioGenerator
from leader_dt.models.communication import UplinkRateModel
from leader_dt.models.aoi import AoiTransitionModel
from leader_dt.models.cpu import CpuBacklogModel
from leader_dt.models.accuracy import AccuracyModel
from leader_dt.rl.observation import ObservationBuilder
from leader_dt.rl.reward import RewardCalculator
from leader_dt.simulator.action import ActionDecoder
from leader_dt.simulator.dynamics import LeaderSynchronizationDynamics
from leader_dt.simulator.recorder import EpisodeRecord, StepRecord

class LeaderSynchronizationEnv(gym.Env):
    """Vehicle-sensor choice upload scheduling environment.

    Action: pair_count scores + one requested accuracy fraction.
    Observation: normalized pair-level AoI, feasibility, data size, and global state.
    """

    metadata = {"render_modes": []}

    def __init__(self, simulation_config: SimulationConfig | None = None) -> None:
        super().__init__()
        self.simulation_config = simulation_config or SimulationConfig()
        self.scenario_generator = ScenarioGenerator(self.simulation_config)
        self.scenario = self.scenario_generator.generate(seed=self.simulation_config.random_seed)
        self.observation_builder = ObservationBuilder(self.simulation_config)
        self.reward_calculator = RewardCalculator(self.simulation_config)
        self.action_decoder = ActionDecoder(pair_count=self.scenario.pair_count)
        self.dynamics = self._build_dynamics()
        self.state = None
        self.episode_record = EpisodeRecord()
        self.random_generator = np.random.default_rng(self.simulation_config.random_seed)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.scenario.pair_count + 1,), dtype=np.float32)
        self.observation_space = self.observation_builder.build_observation_space(self.scenario)

    def _build_dynamics(self) -> LeaderSynchronizationDynamics:
        return LeaderSynchronizationDynamics(
            simulation_config=self.simulation_config,
            scenario=self.scenario,
            uplink_rate_model=UplinkRateModel(self.simulation_config.communication),
            aoi_transition_model=AoiTransitionModel(self.simulation_config.system.freshness_threshold_slots),
            cpu_backlog_model=CpuBacklogModel(self.simulation_config.system.leader_cpu_frequency_cycles_per_second),
            accuracy_model=AccuracyModel(self.simulation_config.system.accuracy_threshold),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.simulation_config.random_seed
        self.random_generator = np.random.default_rng(effective_seed)
        self.scenario = self.scenario_generator.generate(seed=effective_seed)
        self.action_decoder = ActionDecoder(pair_count=self.scenario.pair_count)
        self.dynamics = self._build_dynamics()
        self.state = self.dynamics.initialize_state()
        self.episode_record = EpisodeRecord()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.scenario.pair_count + 1,), dtype=np.float32)
        self.observation_space = self.observation_builder.build_observation_space(self.scenario)
        return self._build_observation(), {}

    def step(self, action: np.ndarray):
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")
        feasible_pair_indices = self.dynamics.get_feasible_pair_indices(self.state)
        slot_index = min(self.state.time_slot_index, self.simulation_config.system.time_horizon_slots - 1)
        scheduling_action = self.action_decoder.decode_rl_action(
            raw_action_array=action,
            feasible_pair_indices=feasible_pair_indices,
            available_data_size_bits_array=self.scenario.available_data_size_bits_matrix[slot_index],
            uplink_capacity_bits_array=self.dynamics.compute_uplink_capacity_bits_array(self.state),
            deterministic=True,
            random_generator=self.random_generator,
        )
        next_state, transition_info = self.dynamics.step(self.state, scheduling_action)
        reward = self.reward_calculator.compute_reward(
            state_after_action=next_state,
            action=scheduling_action,
            achieved_accuracy_float=transition_info["achieved_accuracy_float"],
            priority_weight_array=self.scenario.priority_weight_array_by_pair(),
            weighted_aoi_float=transition_info["weighted_aoi_float"],
            freshness_violation_count_integer=transition_info["freshness_violation_count_integer"],
            accuracy_violation_count_integer=transition_info["accuracy_violation_count_integer"],
            terminal_cpu_violation_count_integer=transition_info["terminal_cpu_violation_count_integer"],
        )
        self.state = next_state
        self.episode_record.append_step(
            StepRecord(
                time_slot_index=next_state.time_slot_index,
                scheduled_pair_index=scheduling_action.scheduled_pair_index,
                collected_bits_float=scheduling_action.collected_bits_float,
                requested_accuracy_fraction_float=scheduling_action.requested_accuracy_fraction_float,
                achieved_accuracy_float=transition_info["achieved_accuracy_float"],
                weighted_aoi_float=transition_info["weighted_aoi_float"],
                cpu_backlog_cycles_float=next_state.cpu_backlog_cycles_float,
                freshness_violation_count=transition_info["freshness_violation_count_integer"],
                accuracy_violation_count=transition_info["accuracy_violation_count_integer"],
                terminal_cpu_violation_count=transition_info["terminal_cpu_violation_count_integer"],
                reward_float=float(reward),
            ),
            next_state.aoi_slots_array,
        )
        terminated = next_state.time_slot_index >= self.simulation_config.system.time_horizon_slots
        truncated = False
        return self._build_observation(), float(reward), terminated, truncated, transition_info

    def _build_observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        feasible_pair_indices = self.dynamics.get_feasible_pair_indices(self.state)
        return self.observation_builder.build_observation(self.state, self.scenario, feasible_pair_indices)
