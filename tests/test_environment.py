import numpy as np
from dataclasses import replace

from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.evaluation.rollout import RolloutRunner
from leader_dt.models.aoi import AoiTransitionModel
from leader_dt.models.cpu import CpuBacklogModel
from leader_dt.models.accuracy import AccuracyModel
from leader_dt.simulator.action import ActionDecoder


def test_environment_reset_shapes():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (env.simulation_config.system.max_pair_count_for_action_space + 1,)


def test_environment_random_step_runs():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)


def test_pair_count_equals_vehicle_count_times_sensors_per_vehicle():
    config = SimulationConfig(random_seed=1)
    env = LeaderSynchronizationEnv(config)
    env.reset(seed=1)
    assert env.scenario.pair_count == config.system.vehicle_count * config.system.sensors_per_vehicle


def test_available_data_size_matrix_shape():
    config = SimulationConfig(random_seed=1)
    env = LeaderSynchronizationEnv(config)
    env.reset(seed=1)
    assert env.scenario.available_data_size_bits_matrix.shape == (
        config.system.time_horizon_slots,
        env.scenario.pair_count,
    )


def test_vehicle_movement_through_zone_b():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    initial_positions = env.state.vehicle_positions_meter_array.copy()
    env.step(env.action_space.sample())
    assert np.any(env.state.vehicle_positions_meter_array != initial_positions)


def test_feasible_pair_mask_matches_in_zone_pairs():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    feasible_pair_indices = env.dynamics.get_feasible_pair_indices(env.state)
    active_pair_indices = env.dynamics.get_active_pair_indices(env.state)
    assert set(feasible_pair_indices).issubset(set(active_pair_indices))
    assert all(0 <= pair_index < env.scenario.pair_count for pair_index in feasible_pair_indices)


def test_aoi_transition_model_refresh_and_growth():
    model = AoiTransitionModel(freshness_threshold_slots=10)
    current = np.array([1.0, 2.0, 3.0])
    next_aoi = model.next_aoi_vector(current, scheduled_pair_index=1, sensing_delay_slots_float=0.5, transmission_delay_slots_float=0.25, refresh_success_boolean=True)
    assert next_aoi[0] == 2.0
    assert next_aoi[1] == 0.75
    assert next_aoi[2] == 4.0


def test_cpu_transition_model():
    model = CpuBacklogModel(cpu_frequency_cycles_per_second=10.0)
    assert model.compute_added_cycles(5.0, 2.0) == 10.0
    assert model.next_backlog_cycles(5.0, 10.0, 1.0) == 5.0


def test_accuracy_calculation():
    model = AccuracyModel(minimum_accuracy_threshold=0.8)
    assert model.compute_accuracy(80.0, 100.0) == 0.8
    assert model.satisfies_accuracy(80.0, 100.0)
    assert not model.satisfies_accuracy(79.0, 100.0)


def test_action_decoder_returns_valid_pair():
    decoder = ActionDecoder(pair_count=3, max_pair_count=5)
    action = np.zeros(6, dtype=np.float32)
    action[1] = 1.0
    action[-1] = 0.8
    scheduling_action = decoder.decode_rl_action(
        raw_action_array=action,
        feasible_pair_indices=[0, 1, 2],
        available_data_size_bits_array=np.array([100.0, 200.0, 300.0]),
        uplink_capacity_bits_array=np.array([100.0, 100.0, 100.0]),
        deterministic=True,
    )
    assert scheduling_action.scheduled_pair_index == 1
    assert scheduling_action.collected_bits_float == 100.0


def test_greedy_policy_action_shape():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    action = GreedyWeightedAoiPolicy().select_action(env)
    assert action.shape == env.action_space.shape


def test_rollout_metrics():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    metrics = RolloutRunner().run_episode(env, GreedyWeightedAoiPolicy(), seed=1)
    assert metrics.average_weighted_aoi_float >= 0.0
    assert metrics.total_collected_bits_float >= 0.0


def test_sensor_type_aoi_state_is_projected_to_pairs():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    assert env.state.sensor_type_aoi_slots_array.shape == (env.scenario.sensor_type_count,)
    projected_pair_aoi = env.scenario.project_sensor_type_values_to_pairs(
        env.state.sensor_type_aoi_slots_array
    )
    assert np.allclose(env.state.aoi_slots_array, projected_pair_aoi)


def test_weighted_aoi_uses_sensor_type_level_state():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    action = GreedyWeightedAoiPolicy().select_action(env)
    _, _, _, _, info = env.step(action)
    active_sensor_type_mask = env.dynamics.get_active_sensor_type_mask(env.state)
    expected_weighted_aoi = env.dynamics.objective.weighted_aoi_at_slot(
        env.state.sensor_type_aoi_slots_array,
        active_sensor_type_mask,
    )
    assert np.isclose(info["weighted_aoi_float"], expected_weighted_aoi)


def _config_with_arrival_rate(arrival_rate: float | None) -> SimulationConfig:
    config = SimulationConfig(random_seed=1)
    return replace(config, data_generation=replace(config.data_generation, sample_arrival_rate_per_slot=arrival_rate))


def test_zero_arrival_rate_leaves_no_feasible_pairs():
    env = LeaderSynchronizationEnv(_config_with_arrival_rate(0.0))
    env.reset(seed=1)
    for _ in range(5):
        assert env.dynamics.get_feasible_pair_indices(env.state) == []
        env.step(env.action_space.sample())


def test_feasible_pairs_require_pending_sample():
    env = LeaderSynchronizationEnv(_config_with_arrival_rate(0.3))
    env.reset(seed=1)
    for _ in range(10):
        pending_sizes = env.dynamics.get_available_data_size_bits_array(env.state)
        for pair_index in env.dynamics.get_feasible_pair_indices(env.state):
            assert pending_sizes[pair_index] > 0.0
        env.step(env.action_space.sample())


def test_upload_consumes_pending_sample_and_ages_aoi():
    env = LeaderSynchronizationEnv(_config_with_arrival_rate(0.3))
    env.reset(seed=1)
    # Wait until some feasible pair holds a sample that is at least one slot old.
    for _ in range(env.simulation_config.system.time_horizon_slots - 1):
        ages = env.dynamics.get_pending_sample_age_slots_array(env.state)
        aged_pairs = [index for index in env.dynamics.get_feasible_pair_indices(env.state) if ages[index] >= 1.0]
        if aged_pairs:
            break
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert aged_pairs
    pair_index = aged_pairs[0]
    sample_age = float(ages[pair_index])
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[pair_index] = 1.0
    action[-1] = 1.0
    arrives_next_slot = env.scenario.sample_arrival_count_matrix[env.state.time_slot_index + 1, pair_index] > 0
    _, _, _, _, info = env.step(action)
    assert info["sample_age_slots_float"] == sample_age
    if not arrives_next_slot:
        assert env.dynamics.get_available_data_size_bits_array(env.state)[pair_index] == 0.0
    if info["refresh_success_boolean"]:
        sensor_type_index = info["scheduled_sensor_type_index"]
        assert env.state.sensor_type_aoi_slots_array[sensor_type_index] >= sample_age


def test_arrival_process_adds_sample_age_observation_block():
    legacy_env = LeaderSynchronizationEnv(_config_with_arrival_rate(None))
    arrival_env = LeaderSynchronizationEnv(_config_with_arrival_rate(0.5))
    max_pair_count = legacy_env.simulation_config.system.max_pair_count_for_action_space
    assert arrival_env.observation_space.shape[0] == legacy_env.observation_space.shape[0] + max_pair_count
    obs, _ = arrival_env.reset(seed=1)
    assert obs.shape == arrival_env.observation_space.shape


def test_large_arrival_rate_matches_legacy_feasibility():
    legacy_env = LeaderSynchronizationEnv(_config_with_arrival_rate(None))
    dense_env = LeaderSynchronizationEnv(_config_with_arrival_rate(50.0))
    legacy_env.reset(seed=1)
    dense_env.reset(seed=1)
    assert legacy_env.dynamics.get_feasible_pair_indices(legacy_env.state) == dense_env.dynamics.get_feasible_pair_indices(dense_env.state)
