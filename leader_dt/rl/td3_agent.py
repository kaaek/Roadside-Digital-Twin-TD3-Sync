"""Stable-Baselines3 TD3 training helpers."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from leader_dt.config import SimulationConfig, Td3TrainingConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class Td3Trainer:
    def __init__(self, simulation_config: SimulationConfig, training_config: Td3TrainingConfig) -> None:
        self.simulation_config = simulation_config
        self.training_config = training_config

    def make_environment(self) -> LeaderSynchronizationEnv:
        return LeaderSynchronizationEnv(self.simulation_config)

    def build_model(self) -> TD3:
        env = DummyVecEnv([self.make_environment])
        action_dimension = env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(action_dimension), sigma=0.2 * np.ones(action_dimension))
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
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.training_config.device,
            seed=self.simulation_config.random_seed,
        )

    def train(self) -> TD3:
        model = self.build_model()
        model.learn(total_timesteps=self.training_config.total_timesteps, progress_bar=False)
        return model

    def save_model(self, model: TD3, output_path: str) -> None:
        model.save(output_path)

    def load_model(self, model_path: str) -> TD3:
        return TD3.load(model_path)
