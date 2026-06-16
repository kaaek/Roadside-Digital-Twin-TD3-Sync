"""Comparison and rollout plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

from leader_dt.evaluation.monte_carlo import MonteCarloResult
from leader_dt.simulator.recorder import EpisodeRecord


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_monte_carlo_comparison_bar(
    monte_carlo_result_dictionary: dict[str, MonteCarloResult],
    metric_name: str = "average_weighted_aoi_float",
    output_path: str | Path = "results/plots/monte_carlo_comparison.png",
    title: str = "Monte Carlo policy comparison",
    ylabel: str | None = None,
) -> Path:
    labels = list(monte_carlo_result_dictionary.keys())
    means = [monte_carlo_result_dictionary[label].metric_mean_dictionary[metric_name] for label in labels]
    stds = [monte_carlo_result_dictionary[label].metric_std_dictionary[metric_name] for label in labels]
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel(ylabel or metric_name)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_policy_consistency_distribution(
    monte_carlo_result_dictionary: dict[str, MonteCarloResult],
    metric_name: str = "average_weighted_aoi_float",
    output_path: str | Path = "results/plots/policy_consistency_distribution.png",
    title: str = "Policy consistency across Monte Carlo trials",
    ylabel: str | None = None,
) -> Path:
    labels = list(monte_carlo_result_dictionary.keys())
    values = [
        [trial_metric_dictionary[metric_name] for trial_metric_dictionary in monte_carlo_result_dictionary[label].per_trial_metric_list]
        for label in labels
    ]
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(values, label=labels, showmeans=True)
    plt.ylabel(ylabel or metric_name)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_aoi_trajectory(
    episode_record: EpisodeRecord,
    output_path: str | Path = "results/plots/aoi_trajectory.png",
    title: str = "AoI trajectory",
    max_pairs_to_plot: int = 20,
) -> Path:
    aoi_history_matrix = episode_record.aoi_history_matrix()
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(10, 5))
    if aoi_history_matrix.size > 0:
        pair_count_to_plot = min(max_pairs_to_plot, aoi_history_matrix.shape[1])
        for pair_index in range(pair_count_to_plot):
            plt.plot(np.arange(1, aoi_history_matrix.shape[0] + 1), aoi_history_matrix[:, pair_index], alpha=0.75)
    plt.xlabel("Time slot")
    plt.ylabel("AoI (slots)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_cpu_backlog(
    episode_record: EpisodeRecord,
    output_path: str | Path = "results/plots/cpu_backlog.png",
    title: str = "CPU backlog trajectory",
) -> Path:
    backlog_values = [record.cpu_backlog_cycles_float for record in episode_record.step_records]
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(backlog_values) + 1), backlog_values, marker="o")
    plt.xlabel("Time slot")
    plt.ylabel("CPU backlog (cycles)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_accuracy_compliance(
    episode_record: EpisodeRecord,
    accuracy_threshold_float: float,
    output_path: str | Path = "results/plots/accuracy_compliance.png",
    title: str = "Upload accuracy compliance",
) -> Path:
    accuracy_values = [record.achieved_accuracy_float for record in episode_record.step_records if not np.isnan(record.achieved_accuracy_float)]
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    if accuracy_values:
        plt.scatter(np.arange(1, len(accuracy_values) + 1), accuracy_values)
    plt.axhline(accuracy_threshold_float, linestyle="--", label="Accuracy threshold")
    plt.xlabel("Refresh event")
    plt.ylabel("Achieved accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path
