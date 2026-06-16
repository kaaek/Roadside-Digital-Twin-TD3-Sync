"""Random policy baseline."""
from __future__ import annotations

from leader_dt.simulator.environment import LeaderSynchronizationEnv

class RandomPolicy:
    def select_action(self, environment: LeaderSynchronizationEnv):
        return environment.action_space.sample()
