"""Evaluation metric dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, fields

from leader_dt.evaluation.penalized_objective import (
    DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS,
    PenalizedObjectiveWeights,
    compute_penalized_score,
)


@dataclass(frozen=True)
class RolloutMetrics:
    average_weighted_aoi_float: float
    maximum_aoi_float: float
    freshness_violation_count_integer: int
    accuracy_violation_count_integer: int
    terminal_cpu_violation_count_integer: int
    final_cpu_backlog_cycles_float: float
    total_collected_bits_float: float
    mean_accuracy_float: float
    episode_return_float: float
    penalized_score_float: float


class MetricCalculator:
    def __init__(
        self,
        penalty_weights: PenalizedObjectiveWeights = DEFAULT_PENALIZED_OBJECTIVE_WEIGHTS,
    ) -> None:
        self.penalty_weights = penalty_weights

    def compute_from_episode_record(self, episode_record) -> RolloutMetrics:
        metrics = dict(episode_record.to_metric_dictionary())

        if "penalized_score_float" not in metrics:
            metrics["penalized_score_float"] = compute_penalized_score(
                average_weighted_aoi_float=metrics["average_weighted_aoi_float"],
                freshness_violation_count_integer=metrics[
                    "freshness_violation_count_integer"
                ],
                terminal_cpu_violation_count_integer=metrics[
                    "terminal_cpu_violation_count_integer"
                ],
                accuracy_violation_count_integer=metrics[
                    "accuracy_violation_count_integer"
                ],
                penalty_weights=self.penalty_weights,
            )

        rollout_metric_field_names = {
            field.name for field in fields(RolloutMetrics)
        }

        filtered_metrics = {
            key: value
            for key, value in metrics.items()
            if key in rollout_metric_field_names
        }

        missing_metric_names = rollout_metric_field_names - filtered_metrics.keys()
        if missing_metric_names:
            raise KeyError(
                "Episode record is missing required rollout metrics: "
                f"{sorted(missing_metric_names)}"
            )

        return RolloutMetrics(**filtered_metrics)