"""Penalty utilities used by evaluation scores and smooth RL rewards.

The raw rollout metrics intentionally remain count-based for reporting.  The
helpers added here are for training-time reward shaping, where threshold jumps
are converted into gradual slack penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PenalizedObjectiveWeights:
    """Penalty weights for converting constraint violations into one comparable score.

    This dataclass is kept for the reported penalized score.  It intentionally
    remains count-based so existing evaluation tables and plots keep the same
    interpretation.
    """

    freshness_violation_weight: float = 10.0
    terminal_cpu_violation_weight: float = 1000.0
    accuracy_violation_weight: float = 100.0


DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS = PenalizedObjectiveWeights()


@dataclass(frozen=True)
class SmoothRewardPenaltyWeights:
    """Weights for smooth training-reward penalties.

    These weights are separate from ``PenalizedObjectiveWeights`` because the
    training reward should be shaped and gradual, while reporting metrics should
    remain directly interpretable.
    """

    freshness_slack_weight: float = 25.0
    accuracy_shortfall_weight: float = 20.0
    cpu_backlog_weight: float = 5.0
    terminal_cpu_backlog_weight: float = 20.0
    softplus_sharpness: float = 5.0


DEFAULT_SMOOTH_REWARD_PENALTY_WEIGHTS = SmoothRewardPenaltyWeights()


def compute_penalized_score(
    average_weighted_aoi_float: float,
    freshness_violation_count_integer: int | float,
    terminal_cpu_violation_count_integer: int | float,
    accuracy_violation_count_integer: int | float,
    penalty_weights: PenalizedObjectiveWeights = DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS,
) -> float:
    """Compute the count-based scalar score used for reporting/evaluation."""
    return float(
        average_weighted_aoi_float
        + penalty_weights.freshness_violation_weight * freshness_violation_count_integer
        + penalty_weights.terminal_cpu_violation_weight * terminal_cpu_violation_count_integer
        + penalty_weights.accuracy_violation_weight * accuracy_violation_count_integer
    )


def squared_positive_slack(slack_float: float) -> float:
    """Return ``max(0, slack)^2`` as a gradual replacement for a hard count."""
    positive_slack = max(0.0, float(slack_float))
    return float(positive_slack * positive_slack)


def softplus_positive_slack(
    slack_float: float,
    sharpness_float: float = DEFAULT_SMOOTH_REWARD_PENALTY_WEIGHTS.softplus_sharpness,
) -> float:
    """Smoothly penalize positive slack with zero penalty at zero slack.

    The subtraction of ``log(2)`` makes the returned penalty equal to zero at a
    zero slack.  Negative slack is clipped because satisfying a constraint should
    not create an additional penalty.
    """
    positive_slack = max(0.0, float(slack_float))
    sharpness = max(float(sharpness_float), 1.0e-12)
    scaled_slack = sharpness * positive_slack
    softplus_value = math.log1p(math.exp(-abs(scaled_slack))) + max(scaled_slack, 0.0)
    return float((softplus_value - math.log(2.0)) / sharpness)


def log1p_backlog_penalty(backlog_ratio_float: float) -> float:
    """Return a smooth CPU-backlog penalty from a nonnegative backlog ratio."""
    return float(math.log1p(max(0.0, float(backlog_ratio_float))))


def add_penalized_score_to_metric_dictionary(
    metric_dictionary: dict,
    penalty_weights: PenalizedObjectiveWeights = DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS,
) -> dict:
    """Return a copy of a metric dictionary with penalized_score_float added."""
    updated_metric_dictionary = dict(metric_dictionary)

    updated_metric_dictionary["penalized_score_float"] = compute_penalized_score(
        average_weighted_aoi_float=updated_metric_dictionary.get(
            "average_weighted_aoi_float",
            0.0,
        ),
        freshness_violation_count_integer=updated_metric_dictionary.get(
            "freshness_violation_count_integer",
            0.0,
        ),
        terminal_cpu_violation_count_integer=updated_metric_dictionary.get(
            "terminal_cpu_violation_count_integer",
            0.0,
        ),
        accuracy_violation_count_integer=updated_metric_dictionary.get(
            "accuracy_violation_count_integer",
            0.0,
        ),
        penalty_weights=penalty_weights,
    )

    updated_metric_dictionary["penalized_objective_freshness_weight"] = (
        penalty_weights.freshness_violation_weight
    )
    updated_metric_dictionary["penalized_objective_terminal_cpu_weight"] = (
        penalty_weights.terminal_cpu_violation_weight
    )
    updated_metric_dictionary["penalized_objective_accuracy_weight"] = (
        penalty_weights.accuracy_violation_weight
    )

    return updated_metric_dictionary
