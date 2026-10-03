"""No-refresh baseline."""
from __future__ import annotations

from leader_dt.simulator.action import PairSchedulingRequest
from leader_dt.simulator.environment import LeaderSynchronizationEnv

class NoRefreshPolicy:
    def select_action(self, environment: LeaderSynchronizationEnv) -> PairSchedulingRequest:
        return PairSchedulingRequest(None, 0.0)
