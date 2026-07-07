"""Policy factory helpers for baseline and Stable-Baselines3 policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from leader_dt import constants
from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy, ProximityGreedyPolicy


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
        td3_model_path: Optional Stable-Baselines3 TD3 checkpoint path.
        ppo_model_path: Optional Stable-Baselines3 PPO checkpoint path.
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

        td3_model = TD3.load(Path(resolved_td3_model_path))
        policy_dictionary["TD3"] = Td3PolicyWrapper(td3_model, deterministic=True)

    if ppo_model_path is not None:
        from stable_baselines3 import PPO
        from leader_dt.rl.wrappers import PpoPolicyWrapper

        ppo_model = PPO.load(Path(ppo_model_path))
        policy_dictionary["PPO"] = PpoPolicyWrapper(ppo_model, deterministic=True)

    return policy_dictionary


def model_path_metadata(
    *,
    legacy_model_path: str | None = None,
    td3_model_path: str | None = None,
    ppo_model_path: str | None = None,
) -> dict[str, str | None]:
    """Return report-friendly model path metadata."""
    resolved_td3_model_path = resolve_td3_model_path(
        legacy_model_path=legacy_model_path,
        td3_model_path=td3_model_path,
    )
    return {
        "td3_model_path": resolved_td3_model_path,
        "ppo_model_path": ppo_model_path,
        "legacy_model_path": legacy_model_path,
    }
