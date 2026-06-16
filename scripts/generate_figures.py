"""Generate figures from saved metric files."""
from __future__ import annotations
from pathlib import Path
import json
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.evaluation.monte_carlo import MonteCarloResult
from leader_dt.plotting.comparison_plots import plot_monte_carlo_comparison_bar, plot_policy_consistency_distribution


def load_monte_carlo_results(path: str | Path) -> dict[str, MonteCarloResult]:
    raw = json.loads(Path(path).read_text())
    results = {}
    for policy_name, payload in raw.items():
        results[policy_name] = MonteCarloResult(
            policy_name=payload["policy_name"],
            metric_mean_dictionary=payload["metric_mean_dictionary"],
            metric_std_dictionary=payload["metric_std_dictionary"],
            per_trial_metric_list=payload["per_trial_metric_list"],
        )
    return results


def main() -> None:
    metrics_path = Path("results/metrics/monte_carlo_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError("Run scripts/run_monte_carlo.py before generating Monte Carlo figures.")
    results = load_monte_carlo_results(metrics_path)
    plot_monte_carlo_comparison_bar(results, output_path="results/plots/monte_carlo_comparison.png")
    plot_policy_consistency_distribution(results, output_path="results/plots/policy_consistency_distribution.png")
    print("Generated Monte Carlo figures.")


if __name__ == "__main__":
    main()
