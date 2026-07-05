"""Stable-Baselines3 PPO construction, monitoring, and checkpoint helpers.

PPO is added as a second RL model while reusing the same simulator,
observation, action, and reward definitions used by TD3. The only PPO-specific
logic lives here: rollout-buffer hyperparameters, clipped policy updates,
logging, and checkpointing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from leader_dt.config import PpoTrainingConfig, SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv


class PpoTrainer:
    """Build, train, save, and load Stable-Baselines3 PPO models."""

    def __init__(self, simulation_config: SimulationConfig, training_config: PpoTrainingConfig) -> None:
        """Store simulator and PPO training configuration objects."""
        self.simulation_config = simulation_config
        self.training_config = training_config

    def make_environment(self, monitor_label: str | None = None) -> Any:
        """Create one simulator environment, optionally wrapped with Monitor.

        Args:
            monitor_label: Optional label used to create a Monitor CSV log file.
                If omitted, the raw ``LeaderSynchronizationEnv`` is returned.

        Returns:
            A raw or monitored Gymnasium-compatible environment.
        """
        environment = LeaderSynchronizationEnv(self.simulation_config)
        if monitor_label is None:
            return environment
        monitor_path = self._monitor_file_path(monitor_label)
        return Monitor(environment, filename=str(monitor_path))

    def build_model(self) -> PPO:
        """Construct a PPO model using the configured hyperparameters."""
        env = DummyVecEnv([lambda: self.make_environment(monitor_label="train")])
        policy_kwargs = dict(
            net_arch=dict(
                pi=list(self.training_config.actor_hidden_layers),
                vf=list(self.training_config.critic_hidden_layers),
            ),
            activation_fn=torch.nn.ReLU,
        )
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            n_epochs=self.training_config.n_epochs,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            clip_range=self.training_config.clip_range,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            tensorboard_log=self.training_config.tensorboard_log_directory,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.training_config.device,
            seed=self.simulation_config.random_seed,
        )

    def train(self) -> PPO:
        """Train PPO for the configured number of timesteps."""
        model = self.build_model()
        callback = self.build_checkpoint_callback()
        model.learn(
            total_timesteps=self.training_config.total_timesteps,
            progress_bar=False,
            callback=callback,
        )
        return model

    def build_checkpoint_callback(self) -> BaseCallback | None:
        """Create a periodic checkpoint callback for standard PPO training."""
        if self.training_config.checkpoint_frequency_steps <= 0:
            return None
        checkpoint_dir = Path(self.training_config.checkpoint_output_directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return CheckpointCallback(
            save_freq=self.training_config.checkpoint_frequency_steps,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    def save_model(self, model: PPO, output_path: str) -> None:
        """Save a PPO model to ``output_path``."""
        model.save(output_path)

    def load_model(self, model_path: str) -> PPO:
        """Load a PPO model from ``model_path``."""
        return PPO.load(model_path)

    def _monitor_file_path(self, monitor_label: str) -> Path:
        """Return the Monitor CSV path for a labelled environment."""
        monitor_dir = Path(self.training_config.monitor_log_directory)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        seed_text = "none" if self.simulation_config.random_seed is None else str(self.simulation_config.random_seed)
        return monitor_dir / f"{monitor_label}_seed_{seed_text}.monitor.csv"
