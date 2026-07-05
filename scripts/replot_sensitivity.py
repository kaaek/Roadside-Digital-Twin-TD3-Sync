"""Replot saved sensitivity JSON files using thesis-quality styling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.evaluation.monte_carlo import MonteCarloResult
from leader_dt.evaluation.sensitivity import SensitivityPointResult
from leader_dt.plotting.sensitivity_plots import plot_sensitivity_curves


def load_sensitivity_results(json_path: Path) -> list[SensitivityPointResult]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    point_results: list[SensitivityPointResult] = []

    for point in payload["points"]:
        policy_results = {}
        for policy_name, policy_payload in point["policy_results"].items():
            policy_results[policy_name] = MonteCarloResult(
                policy_name=policy_payload["policy_name"],
                metric_mean_dictionary=policy_payload["metric_mean_dictionary"],
                metric_std_dictionary=policy_payload["metric_std_dictionary"],
                per_trial_metric_list=policy_payload.get("per_trial_metric_list", []),
            )

        point_results.append(
            SensitivityPointResult(
                parameter_name=point["parameter_name"],
                parameter_value=point["parameter_value"],
                policy_results=policy_results,
            )
        )

    return point_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--metric", default="average_weighted_aoi_float")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    input_json_path = Path(args.input_json)
    sensitivity_results = load_sensitivity_results(input_json_path)

    parameter_name = sensitivity_results[0].parameter_name
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else input_json_path.parent / "plots" / f"sensitivity_{parameter_name}_{args.metric}.png"
    )

    plot_sensitivity_curves(
        sensitivity_result_list=sensitivity_results,
        metric_name=args.metric,
        output_path=output_path,
        title=args.title or f"Sensitivity: {parameter_name}",
    )

    print("Saved:", output_path)
    print("Saved:", output_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()