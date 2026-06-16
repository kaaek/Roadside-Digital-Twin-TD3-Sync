"""Convergence plotting helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_td3_convergence(
    timestep_array,
    reward_array,
    output_path: str | Path = "results/plots/td3_convergence.png",
    title: str = "TD3 training convergence",
    ylabel: str = "Episode reward",
) -> Path:
    timesteps = np.asarray(timestep_array)
    rewards = np.asarray(reward_array, dtype=np.float64)
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, rewards)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_td3_multiseed_convergence(
    seed_to_curve_dictionary: dict[int, tuple],
    output_path: str | Path = "results/plots/td3_multiseed_convergence.png",
    title: str = "TD3 training consistency across seeds",
    ylabel: str = "Evaluation reward",
) -> Path:
    path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4.5))
    for seed_integer, (timestep_array, value_array) in seed_to_curve_dictionary.items():
        plt.plot(timestep_array, value_array, label=f"seed {seed_integer}")
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path
