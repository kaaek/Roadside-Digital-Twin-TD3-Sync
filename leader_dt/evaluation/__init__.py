"""Evaluation helpers for rollouts, Monte Carlo trials, sensitivity, and reporting.

Exports are resolved lazily so importing a dependency-free submodule such as
``leader_dt.evaluation.penalized_objective`` (used by ``leader_dt.rl.reward``)
does not import the simulator environment and create a circular import.
"""
from importlib import import_module

_EXPORT_MODULE_BY_NAME = {
    "ExperimentReport": "leader_dt.evaluation.reporting",
    "MetricCalculator": "leader_dt.evaluation.metrics",
    "MonteCarloEvaluator": "leader_dt.evaluation.monte_carlo",
    "MonteCarloResult": "leader_dt.evaluation.monte_carlo",
    "PenalizedObjectiveWeights": "leader_dt.evaluation.penalized_objective",
    "ReportWriter": "leader_dt.evaluation.reporting",
    "RolloutMetrics": "leader_dt.evaluation.metrics",
    "RolloutRunner": "leader_dt.evaluation.rollout",
    "SensitivityEvaluator": "leader_dt.evaluation.sensitivity",
    "SensitivityPointResult": "leader_dt.evaluation.sensitivity",
}

__all__ = sorted(_EXPORT_MODULE_BY_NAME)


def __getattr__(name: str):
    if name not in _EXPORT_MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULE_BY_NAME[name]), name)
    globals()[name] = value
    return value
