"""Policy wrappers."""
from __future__ import annotations

import numpy as np
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class Td3PolicyWrapper:
    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        observation = environment._build_observation()
        action_array, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action_array, dtype=np.float32)
