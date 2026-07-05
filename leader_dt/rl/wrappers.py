"""Policy wrappers and environment-unwrapping helpers."""
from __future__ import annotations

from typing import Any

import numpy as np

from leader_dt.simulator.environment import LeaderSynchronizationEnv


def unwrap_leader_environment(environment: Any) -> LeaderSynchronizationEnv:
    """Return the underlying ``LeaderSynchronizationEnv`` from wrapper stacks.

    Stable-Baselines3 ``Monitor`` and similar Gymnasium wrappers expose the
    wrapped environment through an ``env`` attribute. Evaluation code and
    handcrafted policies need access to simulator-specific methods such as
    ``_build_observation`` and ``episode_record``, so this helper peels wrappers
    until it reaches the base leader environment.
    """
    current_environment = environment
    visited_object_ids: set[int] = set()
    while not isinstance(current_environment, LeaderSynchronizationEnv):
        object_id = id(current_environment)
        if object_id in visited_object_ids or not hasattr(current_environment, "env"):
            raise TypeError(
                "Could not unwrap environment to LeaderSynchronizationEnv. "
                f"Last object type: {type(current_environment)!r}."
            )
        visited_object_ids.add(object_id)
        current_environment = current_environment.env
    return current_environment


class StableBaselinesPolicyWrapper:
    """Expose any Stable-Baselines3 model through the project policy API.

    The simulator and evaluation modules expect policies to implement
    ``select_action(environment)``. TD3 and PPO both produce continuous actions
    with the same shape, so they can share this wrapper.
    """

    def __init__(self, model: Any, deterministic: bool = True) -> None:
        """Store an SB3 model and deterministic/stochastic prediction mode."""
        self.model = model
        self.deterministic = deterministic

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        """Predict one action from the current simulator observation."""
        base_environment = unwrap_leader_environment(environment)
        observation = base_environment._build_observation()
        action_array, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action_array, dtype=np.float32)


class Td3PolicyWrapper(StableBaselinesPolicyWrapper):
    """Backward-compatible named wrapper for TD3 policies."""


class PpoPolicyWrapper(StableBaselinesPolicyWrapper):
    """Named wrapper for PPO policies."""
