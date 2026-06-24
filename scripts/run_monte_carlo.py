"""Run Monte Carlo evaluation for available policies.

The Greedy baseline used here is CPU-aware by default. Its CPU penalty and
requested accuracy fraction can be overridden from the command line for direct
comparison with TD3 under different heuristic assumptions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt import constants
from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.baselines.no_refresh import NoRefreshPolicy
from leader_dt.baselines.random_policy import RandomPolicy
from leader_dt.config import MonteCarloConfig, SimulationConfig
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator
from leader_dt.evaluation.reporting import ReportWriter
from leader_dt.plotting.comparison_plots import plot_monte_carlo_comparison_bar, plot_policy_consistency_distribution
from leader_dt.simulator.environment import LeaderSynchronizationEnv


def build_policy_dictionary(
    model_path: str | None = None,
    greedy_lambda_cpu: float = constants.DEFAULT_GREEDY_CPU_LAMBDA,
    greedy_requested_accuracy_fraction: float = constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
):
    """Create the policies used for Monte Carlo evaluation."""
    policy_dictionary = {
        "Greedy": GreedyWeightedAoiPolicy(
            lambda_cpu=greedy_lambda_cpu,
            requested_accuracy_fraction=greedy_requested_accuracy_fraction,
        ),
        "No refresh": NoRefreshPolicy(),
        "Random": RandomPolicy(),
    }
    if model_path is not None:
        from stable_baselines3 import TD3
        from leader_dt.rl.wrappers import Td3PolicyWrapper
        model = TD3.load(model_path)
        policy_dictionary["TD3"] = Td3PolicyWrapper(model)
    return policy_dictionary


def main() -> None:
    """Run Monte Carlo evaluation and save metrics, plots, and a report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
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
    monte_carlo_config = MonteCarloConfig(trial_count=args.trials, seed_start=args.seed_start)

    def environment_factory() -> LeaderSynchronizationEnv:
        return LeaderSynchronizationEnv(simulation_config)

    evaluator = MonteCarloEvaluator(trial_count=monte_carlo_config.trial_count, seed_start=monte_carlo_config.seed_start)
    results = evaluator.evaluate_policy_dictionary(
        build_policy_dictionary(
            model_path=args.model_path,
            greedy_lambda_cpu=args.greedy_lambda_cpu,
            greedy_requested_accuracy_fraction=args.greedy_requested_accuracy_fraction,
        ),
        environment_factory,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_writer = ReportWriter()
    metrics_json_path = output_dir / "monte_carlo_metrics.json"
    metrics_csv_path = output_dir / "monte_carlo_metrics.csv"
    # Keep the historical JSON shape: top-level policy names map directly to
    # MonteCarloResult objects. Greedy parameters are stored in the report.
    report_writer.save_metrics_json(results, metrics_json_path)
    report_writer.save_metrics_csv(report_writer.monte_carlo_results_to_rows(results), metrics_csv_path)

    plot_paths = [
        str(plot_monte_carlo_comparison_bar(results, output_path="results/plots/monte_carlo_comparison.png")),
        str(plot_policy_consistency_distribution(results, output_path="results/plots/policy_consistency_distribution.png")),
    ]
    report = report_writer.build_report(
        config_used=simulation_config,
        seed_used=args.seed_start,
        model_path=args.model_path,
        training_hyperparameters={
            "greedy_lambda_cpu": args.greedy_lambda_cpu,
            "greedy_requested_accuracy_fraction": args.greedy_requested_accuracy_fraction,
        },
        metrics_json_path=str(metrics_json_path),
        metrics_csv_path=str(metrics_csv_path),
        plot_paths=plot_paths,
    )
    report_writer.save_report(report, output_dir / "monte_carlo_report.json")
    print("Greedy lambda_cpu:", args.greedy_lambda_cpu)
    print("Greedy requested accuracy fraction:", args.greedy_requested_accuracy_fraction)
    for policy_name, result in results.items():
        print(policy_name, result.metric_mean_dictionary)


if __name__ == "__main__":
    main()
