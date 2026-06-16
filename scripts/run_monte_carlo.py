"""Run Monte Carlo evaluation for available policies."""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.baselines.greedy import GreedyWeightedAoiPolicy
from leader_dt.baselines.no_refresh import NoRefreshPolicy
from leader_dt.baselines.random_policy import RandomPolicy
from leader_dt.config import MonteCarloConfig, SimulationConfig
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator
from leader_dt.evaluation.reporting import ReportWriter
from leader_dt.plotting.comparison_plots import plot_monte_carlo_comparison_bar, plot_policy_consistency_distribution
from leader_dt.simulator.environment import LeaderSynchronizationEnv


def build_policy_dictionary(model_path: str | None = None):
    policy_dictionary = {
        "Greedy": GreedyWeightedAoiPolicy(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/metrics")
    args = parser.parse_args()

    simulation_config = SimulationConfig(random_seed=args.seed_start)
    monte_carlo_config = MonteCarloConfig(trial_count=args.trials, seed_start=args.seed_start)

    def environment_factory() -> LeaderSynchronizationEnv:
        return LeaderSynchronizationEnv(simulation_config)

    evaluator = MonteCarloEvaluator(trial_count=monte_carlo_config.trial_count, seed_start=monte_carlo_config.seed_start)
    results = evaluator.evaluate_policy_dictionary(build_policy_dictionary(args.model_path), environment_factory)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_writer = ReportWriter()
    metrics_json_path = output_dir / "monte_carlo_metrics.json"
    metrics_csv_path = output_dir / "monte_carlo_metrics.csv"
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
        training_hyperparameters={},
        metrics_json_path=str(metrics_json_path),
        metrics_csv_path=str(metrics_csv_path),
        plot_paths=plot_paths,
    )
    report_writer.save_report(report, output_dir / "monte_carlo_report.json")
    for policy_name, result in results.items():
        print(policy_name, result.metric_mean_dictionary)


if __name__ == "__main__":
    main()
