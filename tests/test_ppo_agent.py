"""PPO integration smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from leader_dt.config import PpoTrainingConfig, SimulationConfig, SystemConfig  # noqa: E402
from leader_dt.rl.ppo_agent import PpoTrainer  # noqa: E402
from leader_dt.rl.wrappers import PpoPolicyWrapper, StableBaselinesPolicyWrapper  # noqa: E402
from leader_dt.simulator.environment import LeaderSynchronizationEnv  # noqa: E402


def tiny_simulation_config(seed: int = 1) -> SimulationConfig:
    """Build a small but shape-compatible simulation config for quick tests."""
    return SimulationConfig(
        random_seed=seed,
        system=SystemConfig(
            time_horizon_slots=4,
            vehicle_count=4,
            sensor_type_count=4,
            sensors_per_vehicle=2,
            max_vehicle_count_for_action_space=4,
            max_sensors_per_vehicle_for_action_space=2,
        ),
    )


def test_ppo_model_builds() -> None:
    simulation_config = tiny_simulation_config(seed=1)
    training_config = PpoTrainingConfig(
        total_timesteps=8,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        checkpoint_frequency_steps=0,
    )
    trainer = PpoTrainer(simulation_config, training_config)
    model = trainer.build_model()
    assert model is not None
    assert model.action_space.shape == (simulation_config.system.max_pair_count_for_action_space + 1,)


def test_ppo_wrapper_returns_valid_action_shape() -> None:
    simulation_config = tiny_simulation_config(seed=2)
    training_config = PpoTrainingConfig(
        total_timesteps=8,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        checkpoint_frequency_steps=0,
    )
    env = LeaderSynchronizationEnv(simulation_config)
    env.reset(seed=2)
    model = PpoTrainer(simulation_config, training_config).build_model()
    action = PpoPolicyWrapper(model, deterministic=True).select_action(env)
    assert action.shape == env.action_space.shape
    assert np.all(np.isfinite(action))


def test_generic_sb3_wrapper_is_ppo_compatible() -> None:
    simulation_config = tiny_simulation_config(seed=3)
    training_config = PpoTrainingConfig(
        total_timesteps=8,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        checkpoint_frequency_steps=0,
    )
    env = LeaderSynchronizationEnv(simulation_config)
    env.reset(seed=3)
    model = PpoTrainer(simulation_config, training_config).build_model()
    action = StableBaselinesPolicyWrapper(model, deterministic=True).select_action(env)
    assert action.shape == env.action_space.shape
