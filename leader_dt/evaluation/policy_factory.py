"""Policy factory helpers for baseline and Stable-Baselines3 policies."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from leader_dt import constants
from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy, ProximityGreedyPolicy
from leader_dt.evaluation.monte_carlo import TrainedModelGroup

_TRAINING_SEED_PATTERN = re.compile(r"seed_(\d+)")


def split_model_paths(raw_model_paths: str | None) -> list[str]:
    """Split a comma-separated list of checkpoint paths; ``None`` gives an empty list."""
    if raw_model_paths is None:
        return []
    return [path.strip() for path in raw_model_paths.split(",") if path.strip()]


def training_seeds_from_model_paths(model_paths: list[str]) -> list[int] | None:
    """Read training seeds from ``seed_<k>`` path components; ``None`` if any path lacks one."""
    seeds = []
    for model_path in model_paths:
        matches = _TRAINING_SEED_PATTERN.findall(model_path)
        if not matches:
            return None
        seeds.append(int(matches[-1]))
    return seeds


def _load_model_policy(algorithm_class, wrapper_class, raw_model_paths: str):
    """Load one wrapped model, or a ``TrainedModelGroup`` when several paths are given."""
    model_paths = split_model_paths(raw_model_paths)
    policies = [wrapper_class(algorithm_class.load(Path(model_path)), deterministic=True) for model_path in model_paths]
    if len(policies) == 1:
        return policies[0]
    return TrainedModelGroup(policies=policies, training_seeds=training_seeds_from_model_paths(model_paths))


def resolve_td3_model_path(
    legacy_model_path: str | None = None,
    td3_model_path: str | None = None,
) -> str | None:
    """Resolve the legacy ``--model-path`` alias and the explicit TD3 path.

    ``--model-path`` existed before PPO support and meant a TD3 checkpoint.
    Phase 2 keeps it as a backward-compatible alias for ``--td3-model-path``.
    """
    if legacy_model_path is not None and td3_model_path is not None and legacy_model_path != td3_model_path:
        raise ValueError(
            "Received both --model-path and --td3-model-path with different values. "
            "Use only --td3-model-path, or make both paths identical."
        )
    return td3_model_path or legacy_model_path


def build_policy_dictionary(
    *,
    legacy_model_path: str | None = None,
    td3_model_path: str | None = None,
    ppo_model_path: str | None = None,
    greedy_lambda_cpu: float = constants.DEFAULT_GREEDY_CPU_LAMBDA,
    greedy_requested_accuracy_fraction: float = constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
) -> dict[str, Any]:
    """Create policy dictionaries for Monte Carlo and sensitivity evaluation.

    Args:
        legacy_model_path: Backward-compatible TD3 checkpoint path from the old
            ``--model-path`` CLI argument.
        td3_model_path: Optional Stable-Baselines3 TD3 checkpoint path, or a
            comma-separated list of paths (one per training seed) that are
            evaluated on the same scenarios and pooled into one "TD3" result.
        ppo_model_path: Optional PPO checkpoint path or comma-separated list.
        greedy_lambda_cpu: CPU penalty coefficient for the Greedy baseline.
        greedy_requested_accuracy_fraction: Accuracy fraction requested by the
            Greedy baseline.

    Returns:
        A policy dictionary keyed by human-readable policy names. Plotting code
        iterates over these keys dynamically, so adding Proximity Greedy or PPO
        here automatically adds those policies to Monte Carlo and sensitivity plots.
    """
    resolved_td3_model_path = resolve_td3_model_path(
        legacy_model_path=legacy_model_path,
        td3_model_path=td3_model_path,
    )

    policy_dictionary: dict[str, Any] = {
        "Greedy": GreedyWeightedAoiPolicy(
            lambda_cpu=greedy_lambda_cpu,
            requested_accuracy_fraction=greedy_requested_accuracy_fraction,
        ),
        "Proximity Greedy": ProximityGreedyPolicy(
            lambda_cpu=greedy_lambda_cpu,
            requested_accuracy_fraction=greedy_requested_accuracy_fraction,
        ),
    }

    if resolved_td3_model_path is not None:
        from stable_baselines3 import TD3
        from leader_dt.rl.wrappers import Td3PolicyWrapper

        policy_dictionary["TD3"] = _load_model_policy(TD3, Td3PolicyWrapper, resolved_td3_model_path)

    if ppo_model_path is not None:
        from stable_baselines3 import PPO
        from leader_dt.rl.wrappers import PpoPolicyWrapper

        policy_dictionary["PPO"] = _load_model_policy(PPO, PpoPolicyWrapper, ppo_model_path)

    return policy_dictionary


def model_path_metadata(
    *,
    legacy_model_path: str | None = None,
    td3_model_path: str | None = None,
    ppo_model_path: str | None = None,
) -> dict[str, Any]:
    """Return report-friendly model path metadata, including per-seed path lists."""
    resolved_td3_model_path = resolve_td3_model_path(
        legacy_model_path=legacy_model_path,
        td3_model_path=td3_model_path,
    )
    return {
        "td3_model_path": resolved_td3_model_path,
        "ppo_model_path": ppo_model_path,
        "legacy_model_path": legacy_model_path,
        "td3_model_paths": split_model_paths(resolved_td3_model_path),
        "ppo_model_paths": split_model_paths(ppo_model_path),
    }
