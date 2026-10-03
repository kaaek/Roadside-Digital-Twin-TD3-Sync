"""Diagnose how policies choose pairs over several evaluation scenarios.

For each policy the script reports average weighted AoI, the share of uploads
that fail the accuracy threshold, the mean distance rank of the scheduled pair
among feasible pairs (0 = closest, 0.5 = random), and the share of picks whose
link cannot carry the accuracy threshold of the pending sample in one slot.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace

from leader_dt.baselines.greedy import GreedyMaxAoiPolicy, GreedyWeightedAoiPolicy, ProximityGreedyPolicy
from leader_dt.baselines.no_refresh import NoRefreshPolicy
from leader_dt.baselines.random_policy import RandomPolicy
from leader_dt.config import SimulationConfig
from leader_dt.simulator.environment import LeaderSynchronizationEnv


def build_policy(policy_name: str, td3_model_path: str | None, ppo_model_path: str | None):
    name = policy_name.lower()
    if name == "greedy":
        return GreedyWeightedAoiPolicy()
    if name == "proximity":
        return ProximityGreedyPolicy()
    if name == "maxaoi":
        return GreedyMaxAoiPolicy()
    if name == "random":
        return RandomPolicy()
    if name == "norefresh":
        return NoRefreshPolicy()
    if name == "td3":
        if td3_model_path is None:
            raise ValueError("--td3-model-path is required for TD3 diagnostics.")
        from stable_baselines3 import TD3
        from leader_dt.rl.wrappers import Td3PolicyWrapper
        return Td3PolicyWrapper(TD3.load(td3_model_path, device="cpu"))
    if name == "ppo":
        if ppo_model_path is None:
            raise ValueError("--ppo-model-path is required for PPO diagnostics.")
        from stable_baselines3 import PPO
        from leader_dt.rl.wrappers import PpoPolicyWrapper
        return PpoPolicyWrapper(PPO.load(ppo_model_path, device="cpu"))
    raise ValueError(f"Unsupported policy: {policy_name}")


def diagnose(policy, simulation_config: SimulationConfig, seeds: range) -> dict[str, float]:
    accuracy_threshold = simulation_config.system.accuracy_threshold
    average_aoi_list, distance_rank_list = [], []
    upload_count = failed_upload_count = decision_count = unreachable_pick_count = 0
    for seed in seeds:
        env = LeaderSynchronizationEnv(replace(simulation_config, random_seed=seed))
        env.reset(seed=seed)
        weighted_aoi_sum = 0.0
        done = False
        while not done:
            feasible_pair_indices = env.dynamics.get_feasible_pair_indices(env.state)
            distance_array = env.dynamics.compute_distance_array_by_pair(env.state)
            capacity_array = env.dynamics.compute_uplink_capacity_bits_array(env.state)
            size_array = env.dynamics.get_available_data_size_bits_array(env.state)
            _, _, done, _, info = env.step(policy.select_action(env))
            weighted_aoi_sum += info["weighted_aoi_float"]
            scheduled_pair = env.episode_record.step_records[-1].scheduled_pair_index
            if scheduled_pair is None:
                continue
            upload_count += 1
            failed_upload_count += int(not info["refresh_success_boolean"])
            decision_count += 1
            ordered_pairs = sorted(feasible_pair_indices, key=lambda index: distance_array[index])
            distance_rank_list.append(ordered_pairs.index(scheduled_pair) / max(len(ordered_pairs) - 1, 1))
            unreachable_pick_count += int(capacity_array[scheduled_pair] < accuracy_threshold * size_array[scheduled_pair])
        average_aoi_list.append(weighted_aoi_sum / simulation_config.system.time_horizon_slots)
    return {
        "average_weighted_aoi": float(np.mean(average_aoi_list)),
        "failed_upload_share": failed_upload_count / max(upload_count, 1),
        "mean_distance_rank": float(np.mean(distance_rank_list)) if distance_rank_list else float("nan"),
        "unreachable_pick_share": unreachable_pick_count / max(decision_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=str, default="greedy,proximity,maxaoi,random")
    parser.add_argument("--td3-model-path", type=str, default=None)
    parser.add_argument("--ppo-model-path", type=str, default=None)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--vehicle-count", type=int, default=None)
    args = parser.parse_args()

    simulation_config = SimulationConfig()
    if args.vehicle_count is not None:
        simulation_config = replace(simulation_config, system=replace(simulation_config.system, vehicle_count=args.vehicle_count))
    seeds = range(args.seed_start, args.seed_start + args.trials)

    print(f"{'Policy':<12} {'AoI':>7} {'Failed':>8} {'DistRank':>9} {'Unreachable':>12}")
    for policy_name in [name.strip() for name in args.policies.split(",") if name.strip()]:
        result = diagnose(build_policy(policy_name, args.td3_model_path, args.ppo_model_path), simulation_config, seeds)
        print(
            f"{policy_name:<12} {result['average_weighted_aoi']:>7.2f} {result['failed_upload_share']:>8.0%} "
            f"{result['mean_distance_rank']:>9.2f} {result['unreachable_pick_share']:>12.0%}"
        )


if __name__ == "__main__":
    main()
