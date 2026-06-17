"""Penalized objective used to compare policies under constraint violations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PenalizedObjectiveWeights:
    """Penalty weights for converting constraint violations into one comparable score."""

    freshness_violation_weight: float = 10.0
    terminal_cpu_violation_weight: float = 1000.0
    accuracy_violation_weight: float = 100.0


DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS = PenalizedObjectiveWeights()


def compute_penalized_score(
    average_weighted_aoi_float: float,
    freshness_violation_count_integer: int | float,
    terminal_cpu_violation_count_integer: int | float,
    accuracy_violation_count_integer: int | float,
    penalty_weights: PenalizedObjectiveWeights = DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS,
) -> float:
    """Compute a scalar score that penalizes raw AoI plus constraint violations."""
    return float(
        average_weighted_aoi_float
        + penalty_weights.freshness_violation_weight * freshness_violation_count_integer
        + penalty_weights.terminal_cpu_violation_weight * terminal_cpu_violation_count_integer
        + penalty_weights.accuracy_violation_weight * accuracy_violation_count_integer
    )


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