"""Sensitivity plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from leader_dt.evaluation.sensitivity import SensitivityPointResult
from leader_dt.plotting.thesis_style import apply_thesis_plot_style, thesis_figure_size


POLICY_STYLE_DICTIONARY = {
    "Greedy": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
    "TD3": {"marker": "s", "linestyle": "--", "color": "#009E73"},
    "PPO": {"marker": "^", "linestyle": "-.", "color": "#CC79A7"},
    "No refresh": {"marker": "D", "linestyle": ":", "color": "#D55E00"},
    "Random": {"marker": "v", "linestyle": "-", "color": "#E69F00"},
}


METRIC_LABEL_DICTIONARY = {
    "average_weighted_aoi_float": "Average weighted AoI",
    "maximum_aoi_float": "Maximum AoI",
    "freshness_violation_count_integer": "Freshness violations",
    "accuracy_violation_count_integer": "Accuracy violations",
    "terminal_cpu_violation_count_integer": "Terminal CPU violations",
    "final_cpu_backlog_cycles_float": "Final CPU backlog (cycles)",
    "total_collected_bits_float": "Total collected bits",
    "mean_accuracy_float": "Mean upload accuracy",
    "episode_return_float": "Episode return",
    "penalized_score_float": "Penalized score",
}


PARAMETER_LABEL_DICTIONARY = {
    "sensors_per_vehicle": "Number of sensors per vehicle",
    "accuracy_threshold": "Accuracy threshold",
    "data_size_high_multiplier": "Task-size upper multiplier",
    "vehicle_count": "Number of vehicles",
    "uplink_bandwidth_hz": "Uplink bandwidth (Hz)",
    "freshness_threshold_slots": "Freshness threshold (slots)",
    "zone_size_meter": "Defective-zone size (m)",
}


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _friendly_metric_label(metric_name: str) -> str:
    return METRIC_LABEL_DICTIONARY.get(metric_name, metric_name.replace("_", " "))


def _friendly_parameter_label(parameter_name: str) -> str:
    return PARAMETER_LABEL_DICTIONARY.get(parameter_name, parameter_name.replace("_", " "))


def _save_png_and_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)


def plot_sensitivity_curves(
    sensitivity_result_list: list[SensitivityPointResult],
    metric_name: str = "average_weighted_aoi_float",
    output_path: str | Path = "results/plots/sensitivity_curve.png",
    title: str = "Sensitivity curve",
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Plot thesis-quality sensitivity curves with accessible colors and markers."""
    if len(sensitivity_result_list) == 0:
        raise ValueError("sensitivity_result_list must not be empty.")

    apply_thesis_plot_style()

    policy_names = list(sensitivity_result_list[0].policy_results.keys())
    x_values = np.asarray([point.parameter_value for point in sensitivity_result_list], dtype=float)

    path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=thesis_figure_size("full"))

    for policy_index, policy_name in enumerate(policy_names):
        y_values = np.asarray(
            [
                point.policy_results[policy_name].metric_mean_dictionary[metric_name]
                for point in sensitivity_result_list
            ],
            dtype=float,
        )
        y_errors = np.asarray(
            [
                point.policy_results[policy_name].metric_std_dictionary[metric_name]
                for point in sensitivity_result_list
            ],
            dtype=float,
        )

        style = POLICY_STYLE_DICTIONARY.get(
            policy_name,
            {
                "marker": ["o", "s", "^", "D", "v", "P", "X"][policy_index % 7],
                "linestyle": ["-", "--", "-.", ":"][policy_index % 4],
            },
        )

        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            label=policy_name,
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            elinewidth=0.9,
            capthick=0.9,
            alpha=0.95,
        )

    parameter_name = sensitivity_result_list[0].parameter_name

    ax.set_xlabel(xlabel or _friendly_parameter_label(parameter_name))
    ax.set_ylabel(ylabel or _friendly_metric_label(metric_name))
    ax.set_title(title.replace("_", " "))

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.75)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.45)
    ax.minorticks_on()

    ax.margins(x=0.03)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    _save_png_and_pdf(fig, path)
    plt.close(fig)

    return path