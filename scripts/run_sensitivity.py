"""Run parameter sensitivity experiments."""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.baselines.no_refresh import NoRefreshPolicy
from leader_dt.config import SimulationConfig
from leader_dt.evaluation.reporting import ReportWriter
from leader_dt.evaluation.sensitivity import SensitivityEvaluator
from leader_dt.plotting.sensitivity_plots import plot_sensitivity_curves


def build_policy_dictionary(model_path: str | None = None):
    policy_dictionary = {
        "Greedy": GreedyWeightedAoiPolicy(),
        "No refresh": NoRefreshPolicy(),
    }
    if model_path is not None:
        from stable_baselines3 import TD3
        from leader_dt.rl.wrappers import Td3PolicyWrapper
        model = TD3.load(model_path)
        policy_dictionary["TD3"] = Td3PolicyWrapper(model)
    return policy_dictionary


def parse_values(raw_values: str) -> list[float | int | str]:
    values = []
    for item in raw_values.split(","):
        item = item.strip()
        try:
            value = float(item)
            if value.is_integer():
                value = int(value)
            values.append(value)
        except ValueError:
            values.append(item)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", type=str, default="vehicle_count")
    parser.add_argument("--values", type=str, default="10,20,40,60,80")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/metrics")
    args = parser.parse_args()

    simulation_config = SimulationConfig(random_seed=args.seed_start)
    evaluator = SensitivityEvaluator(simulation_config)
    sensitivity_results = evaluator.run_sweep_with_trials(
        parameter_name=args.parameter,
        parameter_values=parse_values(args.values),
        policy_dictionary=build_policy_dictionary(args.model_path),
        trial_count=args.trials,
        seed_start=args.seed_start,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_writer = ReportWriter()
    metrics_json_path = output_dir / f"sensitivity_{args.parameter}.json"
    report_writer.save_metrics_json({"points": sensitivity_results}, metrics_json_path)
    plot_path = plot_sensitivity_curves(
        sensitivity_results,
        output_path=f"results/plots/sensitivity_{args.parameter}.png",
        title=f"Sensitivity: {args.parameter}",
    )
    report = report_writer.build_report(
        config_used=simulation_config,
        seed_used=args.seed_start,
        model_path=args.model_path,
        metrics_json_path=str(metrics_json_path),
        plot_paths=[str(plot_path)],
    )
    report_writer.save_report(report, output_dir / f"sensitivity_{args.parameter}_report.json")
    print("Saved", metrics_json_path)
    print("Saved", plot_path)


if __name__ == "__main__":
    main()
