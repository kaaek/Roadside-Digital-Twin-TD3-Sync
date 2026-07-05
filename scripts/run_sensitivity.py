"""Run parameter sensitivity experiments for baselines, TD3, and PPO.

The script supports optional command-line overrides for the CPU-aware Greedy
policy. It keeps the historical ``--model-path`` argument as a TD3 checkpoint
alias while adding explicit ``--td3-model-path`` and ``--ppo-model-path``
arguments for multi-RL-model sensitivity plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt import constants
from leader_dt.config import SimulationConfig
from leader_dt.evaluation.policy_factory import build_policy_dictionary, model_path_metadata
from leader_dt.evaluation.reporting import ReportWriter
from leader_dt.evaluation.sensitivity import SensitivityEvaluator
from leader_dt.plotting.sensitivity_plots import plot_sensitivity_curves

def parse_values(raw_values: str) -> list[float | int | str]:
    """Parse comma-separated CLI values into int, float, or string values."""
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
    """Run one sensitivity sweep and save metrics, plots, and report files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", type=str, default="vehicle_count")
    parser.add_argument("--values", type=str, default="10,20,40,60,80")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Backward-compatible alias for --td3-model-path.",
    )
    parser.add_argument(
        "--td3-model-path",
        type=str,
        default=None,
        help="Optional Stable-Baselines3 TD3 checkpoint path.",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default=None,
        help="Optional Stable-Baselines3 PPO checkpoint path.",
    )
    parser.add_argument("--output-dir", type=str, default="results/metrics")
    parser.add_argument(
        "--greedy-lambda-cpu",
        type=float,
        default=constants.DEFAULT_GREEDY_CPU_LAMBDA,
        help="CPU penalty coefficient for the CPU-aware Greedy baseline.",
    )
    parser.add_argument(
        "--greedy-requested-accuracy-fraction",
        type=float,
        default=constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
        help="Payload fraction requested by the CPU-aware Greedy baseline.",
    )
    args = parser.parse_args()

    simulation_config = SimulationConfig(random_seed=args.seed_start)
    evaluator = SensitivityEvaluator(simulation_config)
    sensitivity_results = evaluator.run_sweep_with_trials(
        parameter_name=args.parameter,
        parameter_values=parse_values(args.values),
        policy_dictionary=build_policy_dictionary(
            legacy_model_path=args.model_path,
            td3_model_path=args.td3_model_path,
            ppo_model_path=args.ppo_model_path,
            # include_random_policy=False,
            greedy_lambda_cpu=args.greedy_lambda_cpu,
            greedy_requested_accuracy_fraction=args.greedy_requested_accuracy_fraction,
        ),
        trial_count=args.trials,
        seed_start=args.seed_start,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_writer = ReportWriter()
    metrics_json_path = output_dir / f"sensitivity_{args.parameter}.json"
    report_writer.save_metrics_json(
        {
            **model_path_metadata(
                legacy_model_path=args.model_path,
                td3_model_path=args.td3_model_path,
                ppo_model_path=args.ppo_model_path,
            ),
            "greedy_lambda_cpu": args.greedy_lambda_cpu,
            "greedy_requested_accuracy_fraction": args.greedy_requested_accuracy_fraction,
            "points": sensitivity_results,
        },
        metrics_json_path,
    )
    plot_output_dir = output_dir / "plots"
    plot_path = plot_sensitivity_curves(
        sensitivity_results,
        output_path=plot_output_dir / f"sensitivity_{args.parameter}.png",
        title=f"Sensitivity: {args.parameter}",
    )
    report = report_writer.build_report(
        config_used=simulation_config,
        seed_used=args.seed_start,
        model_path=args.td3_model_path or args.model_path or args.ppo_model_path,
        training_hyperparameters={
            **model_path_metadata(
                legacy_model_path=args.model_path,
                td3_model_path=args.td3_model_path,
                ppo_model_path=args.ppo_model_path,
            ),
            "greedy_lambda_cpu": args.greedy_lambda_cpu,
            "greedy_requested_accuracy_fraction": args.greedy_requested_accuracy_fraction,
        },
        metrics_json_path=str(metrics_json_path),
        plot_paths=[str(plot_path)],
    )
    report_writer.save_report(report, output_dir / f"sensitivity_{args.parameter}_report.json")
    print("TD3 model path:", args.td3_model_path or args.model_path)
    print("PPO model path:", args.ppo_model_path)
    print("Greedy lambda_cpu:", args.greedy_lambda_cpu)
    print("Greedy requested accuracy fraction:", args.greedy_requested_accuracy_fraction)
    print("Saved", metrics_json_path)
    print("Saved", plot_path)


if __name__ == "__main__":
    main()
