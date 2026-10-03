import numpy as np
from dataclasses import replace

from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.baselines.greedy import GreedyMaxAoiPolicy, GreedyWeightedAoiPolicy, ProximityGreedyPolicy
from leader_dt.evaluation.rollout import RolloutRunner
from leader_dt.models.aoi import AoiTransitionModel
from leader_dt.models.cpu import CpuBacklogModel
from leader_dt.models.accuracy import AccuracyModel
from leader_dt.rl.pair_features import PAIR_FEATURE_COUNT, PAIR_FEATURE_NAMES
from leader_dt.simulator.action import ActionDecoder, PairSchedulingRequest


def test_environment_reset_shapes():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (PAIR_FEATURE_COUNT + 1,)


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


def test_greedy_policy_returns_feasible_request():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    request = GreedyWeightedAoiPolicy().select_action(env)
    assert isinstance(request, PairSchedulingRequest)
    assert request.pair_index in env.dynamics.get_feasible_pair_indices(env.state)


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
        env.step(PairSchedulingRequest(None, 0.0))
    assert aged_pairs
    pair_index = aged_pairs[0]
    sample_age = float(ages[pair_index])
    arrives_next_slot = env.scenario.sample_arrival_count_matrix[env.state.time_slot_index + 1, pair_index] > 0
    _, _, _, _, info = env.step(PairSchedulingRequest(pair_index, 1.0))
    assert info["sample_age_slots_float"] == sample_age
    if not arrives_next_slot:
        assert env.dynamics.get_available_data_size_bits_array(env.state)[pair_index] == 0.0
    if info["refresh_success_boolean"]:
        sensor_type_index = info["scheduled_sensor_type_index"]
        assert env.state.sensor_type_aoi_slots_array[sensor_type_index] >= sample_age


def test_large_arrival_rate_matches_legacy_feasibility():
    legacy_env = LeaderSynchronizationEnv(_config_with_arrival_rate(None))
    dense_env = LeaderSynchronizationEnv(_config_with_arrival_rate(50.0))
    legacy_env.reset(seed=1)
    dense_env.reset(seed=1)
    assert legacy_env.dynamics.get_feasible_pair_indices(legacy_env.state) == dense_env.dynamics.get_feasible_pair_indices(dense_env.state)


def test_action_decoder_picks_highest_weighted_feature_score():
    decoder = ActionDecoder(feature_count=2)
    pair_feature_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # Weight action 0.0 -> w = -1, 1.0 -> w = +1: prefer feature 1, avoid feature 0.
    scheduling_action = decoder.decode_rl_action(
        raw_action_array=np.array([0.0, 1.0, 1.0]),
        feasible_pair_indices=[0, 1, 2],
        pair_feature_matrix=pair_feature_matrix,
        available_data_size_bits_array=np.array([100.0, 200.0, 300.0]),
        uplink_capacity_bits_array=np.array([500.0, 150.0, 500.0]),
    )
    assert scheduling_action.scheduled_pair_index == 1
    assert scheduling_action.collected_bits_float == 150.0
    # Infeasible pairs are never chosen, even with the best score.
    scheduling_action = decoder.decode_rl_action(
        raw_action_array=np.array([0.0, 1.0, 1.0]),
        feasible_pair_indices=[0, 2],
        pair_feature_matrix=pair_feature_matrix,
        available_data_size_bits_array=np.array([100.0, 200.0, 300.0]),
        uplink_capacity_bits_array=np.array([500.0, 150.0, 500.0]),
    )
    assert scheduling_action.scheduled_pair_index == 2


def test_observation_is_compact_and_vehicle_count_independent():
    shapes = set()
    for vehicle_count in (10, 40, 80):
        config = SimulationConfig(random_seed=1)
        config = replace(config, system=replace(config.system, vehicle_count=vehicle_count))
        env = LeaderSynchronizationEnv(config)
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert np.all((obs >= 0.0) & (obs <= 1.0))
        shapes.add(obs.shape)
    assert shapes == {(80,)}


def test_pair_features_are_bounded_and_reachable_accuracy_matches_capacity():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    features = env.build_pair_feature_matrix()
    assert features.shape == (env.scenario.pair_count, PAIR_FEATURE_COUNT)
    assert np.all(features >= 0.0) and np.all(features <= 3.0)
    sizes = env.dynamics.get_available_data_size_bits_array(env.state)
    capacity = env.dynamics.compute_uplink_capacity_bits_array(env.state)
    column = PAIR_FEATURE_NAMES.index("reachable_accuracy")
    with_sample = sizes > 0.0
    assert np.allclose(features[with_sample, column], np.minimum(1.0, capacity[with_sample] / sizes[with_sample]))
    assert np.all(features[~with_sample, column] == 0.0)



def _initial_positions(env: LeaderSynchronizationEnv) -> np.ndarray:
    return env.scenario.initial_vehicle_positions_meter_array.copy()


def test_unseeded_reset_draws_a_new_scenario():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    first_positions = _initial_positions(env)
    env.reset()
    assert not np.allclose(first_positions, _initial_positions(env))


def test_seeded_reset_is_reproducible_across_instances():
    first_env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    second_env = LeaderSynchronizationEnv(SimulationConfig(random_seed=7))
    first_env.reset()
    first_env.reset(seed=42)
    second_env.reset(seed=42)
    assert np.allclose(_initial_positions(first_env), _initial_positions(second_env))


def test_vec_env_auto_resets_follow_a_reproducible_scenario_sequence():
    from stable_baselines3.common.vec_env import DummyVecEnv

    def collect_positions(training_seed: int) -> list[np.ndarray]:
        vec_env = DummyVecEnv([lambda: LeaderSynchronizationEnv(SimulationConfig(random_seed=training_seed))])
        vec_env.seed(training_seed)
        vec_env.reset()
        base_env = vec_env.envs[0]
        positions = [_initial_positions(base_env)]
        action = np.zeros((1, *vec_env.action_space.shape), dtype=np.float32)
        for _ in range(2):
            done = False
            while not done:
                _, _, dones, _ = vec_env.step(action)
                done = bool(dones[0])
            positions.append(_initial_positions(base_env))
        return positions

    first_run = collect_positions(training_seed=1)
    second_run = collect_positions(training_seed=1)
    assert not np.allclose(first_run[0], first_run[1])
    assert not np.allclose(first_run[1], first_run[2])
    for first, second in zip(first_run, second_run):
        assert np.allclose(first, second)
    assert not np.allclose(first_run[0], collect_positions(training_seed=2)[0])
