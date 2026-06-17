"""Generate a bar chart comparing raw AoI and penalized score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def extract_policy_means(result_object: dict) -> dict[str, dict]:
    if "metric_mean_dictionary" in result_object:
        return {
            result_object["policy_name"]: result_object["metric_mean_dictionary"],
        }

    extracted_dictionary = {}

    for policy_name, policy_result in result_object.items():
        if isinstance(policy_result, dict) and "metric_mean_dictionary" in policy_result:
            extracted_dictionary[policy_name] = policy_result["metric_mean_dictionary"]
        elif isinstance(policy_result, dict) and "average_weighted_aoi_float" in policy_result:
            extracted_dictionary[policy_name] = policy_result

    return extracted_dictionary


def plot_metric_bar(
    policy_metric_dictionary: dict[str, dict],
    metric_name: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    policy_names = list(policy_metric_dictionary.keys())
    metric_values = [
        policy_metric_dictionary[policy_name][metric_name]
        for policy_name in policy_names
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(policy_names, metric_values)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/metrics/monte_carlo_results_penalized.json",
    )
    parser.add_argument(
        "--raw-output",
        default="results/plots/monte_carlo_raw_aoi_comparison.png",
    )
    parser.add_argument(
        "--penalized-output",
        default="results/plots/monte_carlo_penalized_score_comparison.png",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    result_object = json.loads(input_path.read_text())
    policy_metric_dictionary = extract_policy_means(result_object)

    plot_metric_bar(
        policy_metric_dictionary=policy_metric_dictionary,
        metric_name="average_weighted_aoi_float",
        output_path=Path(args.raw_output),
        title="Monte Carlo Raw Weighted AoI Comparison",
        y_label="Average weighted AoI",
    )

    plot_metric_bar(
        policy_metric_dictionary=policy_metric_dictionary,
        metric_name="penalized_score_float",
        output_path=Path(args.penalized_output),
        title="Monte Carlo Penalized Score Comparison",
        y_label="Penalized score",
    )

    print(f"Saved {args.raw_output}")
    print(f"Saved {args.penalized_output}")


if __name__ == "__main__":
    main()