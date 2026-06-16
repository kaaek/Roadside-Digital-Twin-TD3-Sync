from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv


def test_environment_reset_shapes():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (env.scenario.pair_count + 1,)


def test_environment_random_step_runs():
    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=1))
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
