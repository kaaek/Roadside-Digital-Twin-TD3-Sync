"""No-refresh baseline."""
from __future__ import annotations

import numpy as np
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class NoRefreshPolicy:
    def select_action(self, environment: LeaderSynchronizationEnv) -> np.ndarray:
        return np.zeros(environment.action_space.shape, dtype=np.float32)
