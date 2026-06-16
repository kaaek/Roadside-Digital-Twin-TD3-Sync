"""Fast smoke tests for the simulator."""
from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv
from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.evaluation.rollout import RolloutRunner


def main() -> None:
    config = SimulationConfig(random_seed=1)
    env = LeaderSynchronizationEnv(config)
    obs, _ = env.reset(seed=1)
    print("Observation shape:", obs.shape)
    print("Action shape:", env.action_space.shape)
    print("Pair count:", env.scenario.pair_count)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (env.simulation_config.system.max_pair_count_for_action_space + 1,)

    policy = GreedyWeightedAoiPolicy()
    metrics = RolloutRunner().run_episode(env, policy, seed=1)
    print("Smoke rollout metrics:", metrics)
    assert metrics.average_weighted_aoi_float > 0
    assert env.state.time_slot_index == config.system.time_horizon_slots
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
