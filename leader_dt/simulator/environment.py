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
from leader_dt.rl.pair_features import PAIR_FEATURE_COUNT, PairFeatureBuilder
from leader_dt.rl.reward import RewardCalculator
from leader_dt.simulator.action import ActionDecoder, PairSchedulingRequest, build_scheduling_action
from leader_dt.simulator.dynamics import LeaderSynchronizationDynamics
from leader_dt.simulator.recorder import EpisodeRecord, StepRecord

class LeaderSynchronizationEnv(gym.Env):
    """Vehicle-sensor choice upload scheduling environment.

    Action: one weight per pair feature plus one requested accuracy fraction; the
    feasible pair with the highest weighted feature score is scheduled.  Heuristic
    baselines pass a ``PairSchedulingRequest`` to ``step`` instead.
    Observation: see ``ObservationBuilder``.
    Seeding: ``reset(seed=s)`` regenerates the scenario deterministically from
    ``s``; ``reset()`` draws the next scenario from the same random stream.
    """

    metadata = {"render_modes": []}

    def __init__(self, simulation_config: SimulationConfig | None = None) -> None:
        super().__init__()
        self.simulation_config = simulation_config or SimulationConfig()
        self.scenario_generator = ScenarioGenerator(self.simulation_config)
        self.scenario = self.scenario_generator.generate(seed=self.simulation_config.random_seed)
        self.observation_builder = ObservationBuilder(self.simulation_config)
        self.reward_calculator = RewardCalculator(self.simulation_config)
        self.pair_feature_builder = PairFeatureBuilder(self.simulation_config)
        self.action_decoder = ActionDecoder(feature_count=PAIR_FEATURE_COUNT)
        self.dynamics = self._build_dynamics()
        self.state = None
        self.episode_record = EpisodeRecord()
        self.random_generator = np.random.default_rng(self.simulation_config.random_seed)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_decoder.action_dimension,), dtype=np.float32)
        self.observation_space = self.observation_builder.build_observation_space(self.scenario)

    def build_pair_feature_matrix(self) -> np.ndarray:
        """Per-pair features for the current state (see ``leader_dt/rl/pair_features.py``)."""
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        return self.pair_feature_builder.build(self.dynamics, self.state)

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
        # An explicit seed reseeds the scenario stream; ``None`` (e.g. an SB3
        # auto-reset) continues it, so consecutive episodes draw new scenarios.
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)
        self.scenario = self.scenario_generator.generate(seed=seed)
        self.dynamics = self._build_dynamics()
        self.state = self.dynamics.initialize_state()
        self.episode_record = EpisodeRecord()
        return self._build_observation(), {}

    def _decode_action(self, action: np.ndarray | PairSchedulingRequest):
        feasible_pair_indices = self.dynamics.get_feasible_pair_indices(self.state)
        available_data_size_bits_array = self.dynamics.get_available_data_size_bits_array(self.state)
        uplink_capacity_bits_array = self.dynamics.compute_uplink_capacity_bits_array(self.state)
        if isinstance(action, PairSchedulingRequest):
            return build_scheduling_action(
                action.pair_index,
                action.requested_accuracy_fraction,
                feasible_pair_indices,
                available_data_size_bits_array,
                uplink_capacity_bits_array,
            )
        return self.action_decoder.decode_rl_action(
            raw_action_array=action,
            feasible_pair_indices=feasible_pair_indices,
            pair_feature_matrix=self.build_pair_feature_matrix(),
            available_data_size_bits_array=available_data_size_bits_array,
            uplink_capacity_bits_array=uplink_capacity_bits_array,
        )

    def step(self, action: np.ndarray | PairSchedulingRequest):
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")
        scheduling_action = self._decode_action(action)
        next_state, transition_info = self.dynamics.step(self.state, scheduling_action)
        reward = self.reward_calculator.compute_reward(
            state_after_action=next_state,
            action=scheduling_action,
            achieved_accuracy_float=transition_info["achieved_accuracy_float"],
            priority_weight_array=self.scenario.priority_weight_array_by_sensor_type(),
            weighted_aoi_float=transition_info["weighted_aoi_float"],
            freshness_violation_count_integer=transition_info["freshness_violation_count_integer"],
            accuracy_violation_count_integer=transition_info["accuracy_violation_count_integer"],
            terminal_cpu_violation_count_integer=transition_info["terminal_cpu_violation_count_integer"],
        )
        self.state = next_state
        active_pair_mask_array = self.dynamics.get_active_pair_mask(next_state)
        active_sensor_type_mask_array = self.dynamics.get_active_sensor_type_mask(next_state)
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
                active_pair_count=transition_info.get("active_pair_count_integer", int(np.sum(active_pair_mask_array))),
                active_sensor_type_count=transition_info.get("active_sensor_type_count_integer", int(np.sum(active_sensor_type_mask_array))),
            ),
            next_state.sensor_type_aoi_slots_array,
            active_sensor_type_mask_array,
        )
        terminated = next_state.time_slot_index >= self.simulation_config.system.time_horizon_slots
        truncated = False
        return self._build_observation(), float(reward), terminated, truncated, transition_info

    def _build_observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        feasible_pair_indices = self.dynamics.get_feasible_pair_indices(self.state)
        active_pair_indices = self.dynamics.get_active_pair_indices(self.state)
        return self.observation_builder.build_observation(self.state, self.scenario, feasible_pair_indices, active_pair_indices, self.build_pair_feature_matrix())
