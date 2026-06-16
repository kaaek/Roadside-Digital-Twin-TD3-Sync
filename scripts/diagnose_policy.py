"""Diagnose a policy on one rollout."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv


def build_policy(policy_name: str, model_path: str | None = None):
    if policy_name.lower() == "greedy":
        return GreedyWeightedAoiPolicy()
    if policy_name.lower() == "td3":
        if model_path is None:
            raise ValueError("--model-path is required for TD3 diagnostics.")
        from stable_baselines3 import TD3
        from leader_dt.rl.wrappers import Td3PolicyWrapper
        return Td3PolicyWrapper(TD3.load(model_path))
    raise ValueError(f"Unsupported policy: {policy_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="greedy")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = LeaderSynchronizationEnv(SimulationConfig(random_seed=args.seed))
    policy = build_policy(args.policy, args.model_path)
    observation, _ = env.reset(seed=args.seed)
    done = False
    selected_pair_list = []
    requested_accuracy_list = []
    while not done:
        action = policy.select_action(env)
        selected_pair_list.append(int(np.argmax(action[:-1])) if action[:-1].size else -1)
        requested_accuracy_list.append(float(action[-1]))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    print("Policy:", args.policy)
    print("Selected pair counts:")
    unique_values, counts = np.unique(selected_pair_list, return_counts=True)
    for pair_index, count in zip(unique_values, counts):
        print(int(pair_index), int(count))
    print("Requested accuracy mean:", float(np.mean(requested_accuracy_list)))
    print("Metrics:", env.episode_record.to_metric_dictionary())


if __name__ == "__main__":
    main()
