"""Stable-Baselines3 TD3 construction, monitoring, and checkpoint helpers.

This module centralizes TD3-specific experiment settings so the scripts can
vary action noise, target-policy smoothing, replay-buffer size, batch size,
TensorBoard logging, Monitor logging, and periodic checkpointing without
changing the simulator code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from leader_dt.config import SimulationConfig, Td3TrainingConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv


class Td3Trainer:
    """Build, train, save, and load Stable-Baselines3 TD3 models."""

    def __init__(self, simulation_config: SimulationConfig, training_config: Td3TrainingConfig) -> None:
        """Store simulator and TD3 training configuration objects."""
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

    def build_model(self) -> TD3:
        """Construct a TD3 model using the configured hyperparameters."""
        env = DummyVecEnv([lambda: self.make_environment(monitor_label="train")])
        action_dimension = env.action_space.shape[-1]
        noise = NormalActionNoise(
            mean=np.zeros(action_dimension),
            sigma=self.training_config.action_noise_sigma * np.ones(action_dimension),
        )
        policy_kwargs = dict(
            net_arch=dict(
                pi=list(self.training_config.actor_hidden_layers),
                qf=list(self.training_config.critic_hidden_layers),
            ),
            activation_fn=torch.nn.ReLU,
        )
        return TD3(
            "MlpPolicy",
            env,
            action_noise=noise,
            learning_rate=self.training_config.learning_rate,
            learning_starts=self.training_config.learning_starts,
            buffer_size=self.training_config.buffer_size,
            batch_size=self.training_config.batch_size,
            tau=self.training_config.tau,
            gamma=self.training_config.gamma,
            policy_delay=self.training_config.policy_delay,
            train_freq=(self.training_config.train_frequency_steps, "step"),
            gradient_steps=self.training_config.gradient_steps,
            target_policy_noise=self.training_config.target_policy_noise,
            target_noise_clip=self.training_config.target_noise_clip,
            tensorboard_log=self.training_config.tensorboard_log_directory,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.training_config.device,
            seed=self.simulation_config.random_seed,
        )

    def train(self) -> TD3:
        """Train TD3 for the configured number of timesteps.

        Periodic checkpoints are saved when ``checkpoint_frequency_steps`` is
        positive. Best/latest checkpoint management for convergence experiments
        is handled by ``scripts/train_td3_until_convergence.py``.
        """
        model = self.build_model()
        callback = self.build_checkpoint_callback()
        model.learn(
            total_timesteps=self.training_config.total_timesteps,
            progress_bar=False,
            callback=callback,
        )
        return model

    def build_checkpoint_callback(self) -> BaseCallback | None:
        """Create a periodic checkpoint callback for standard TD3 training."""
        if self.training_config.checkpoint_frequency_steps <= 0:
            return None
        checkpoint_dir = Path(self.training_config.checkpoint_output_directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return CheckpointCallback(
            save_freq=self.training_config.checkpoint_frequency_steps,
            save_path=str(checkpoint_dir),
            name_prefix="td3_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    def save_model(self, model: TD3, output_path: str) -> None:
        """Save a TD3 model to ``output_path``."""
        model.save(output_path)

    def load_model(self, model_path: str) -> TD3:
        """Load a TD3 model from ``model_path``."""
        return TD3.load(model_path)

    def _monitor_file_path(self, monitor_label: str) -> Path:
        """Return the Monitor CSV path for a labelled environment."""
        monitor_dir = Path(self.training_config.monitor_log_directory)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        seed_text = "none" if self.simulation_config.random_seed is None else str(self.simulation_config.random_seed)
        return monitor_dir / f"{monitor_label}_seed_{seed_text}.monitor.csv"
