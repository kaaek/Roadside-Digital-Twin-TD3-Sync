"""Sensitivity plotting helpers."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from leader_dt.evaluation.sensitivity import SensitivityPointResult


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_sensitivity_curves(
    sensitivity_result_list: list[SensitivityPointResult],
    metric_name: str = "average_weighted_aoi_float",
    output_path: str | Path = "results/plots/sensitivity_curve.png",
    title: str = "Sensitivity curve",
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    if len(sensitivity_result_list) == 0:
        raise ValueError("sensitivity_result_list must not be empty.")
    policy_names = list(sensitivity_result_list[0].policy_results.keys())
    x_values = [point.parameter_value for point in sensitivity_result_list]
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    for policy_name in policy_names:
        y_values = [point.policy_results[policy_name].metric_mean_dictionary[metric_name] for point in sensitivity_result_list]
        y_errors = [point.policy_results[policy_name].metric_std_dictionary[metric_name] for point in sensitivity_result_list]
        plt.errorbar(x_values, y_values, yerr=y_errors, marker="o", capsize=3, label=policy_name)
    plt.xlabel(xlabel or sensitivity_result_list[0].parameter_name)
    plt.ylabel(ylabel or metric_name)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path
