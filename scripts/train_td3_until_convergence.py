"""Train TD3 with periodic evaluation and convergence-based early stopping.

This script is intended for the long-run TD3 experiment: it trains in chunks,
evaluates the policy every configured number of timesteps, saves the best model,
records evaluation history, and stops when the evaluation reward stops improving
for a configured patience window after the minimum training budget is reached.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.config import (  # noqa: E402
    SimulationConfig,
    Td3ConvergenceTrainingConfig,
    Td3TrainingConfig,
)
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator  # noqa: E402
from leader_dt.evaluation.reporting import ReportWriter  # noqa: E402
from leader_dt.plotting.convergence_plots import plot_td3_convergence  # noqa: E402
from leader_dt.rl.td3_agent import Td3Trainer  # noqa: E402
from leader_dt.rl.wrappers import Td3PolicyWrapper  # noqa: E402
from leader_dt.simulator.environment import LeaderSynchronizationEnv  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    convergence_defaults = Td3ConvergenceTrainingConfig()
    training_defaults = Td3TrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train TD3 until evaluation reward converges, while saving the "
            "best model, latest model, evaluation history, and convergence plots."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--eval-frequency-steps",
        type=int,
        default=convergence_defaults.eval_frequency_steps,
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=convergence_defaults.evaluation_episode_count,
    )
    parser.add_argument(
        "--patience-evaluations",
        type=int,
        default=convergence_defaults.patience_evaluation_count,
    )
    parser.add_argument(
        "--minimum-timesteps",
        type=int,
        default=convergence_defaults.minimum_training_timesteps,
    )
    parser.add_argument(
        "--maximum-timesteps",
        type=int,
        default=convergence_defaults.maximum_training_timesteps,
    )
    parser.add_argument(
        "--minimum-reward-improvement",
        type=float,
        default=convergence_defaults.minimum_reward_improvement_float,
    )
    parser.add_argument(
        "--evaluation-seed-start",
        type=int,
        default=convergence_defaults.evaluation_seed_start,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=convergence_defaults.output_directory,
    )
    parser.add_argument(
        "--best-model-name",
        type=str,
        default=convergence_defaults.best_model_name,
    )
    parser.add_argument(
        "--latest-model-name",
        type=str,
        default=convergence_defaults.latest_model_name,
    )
    parser.add_argument(
        "--sb3-log-interval",
        type=int,
        default=convergence_defaults.sb3_log_interval,
    )
    parser.add_argument("--learning-rate", type=float, default=training_defaults.learning_rate)
    parser.add_argument("--learning-starts", type=int, default=training_defaults.learning_starts)
    parser.add_argument("--buffer-size", type=int, default=training_defaults.buffer_size)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument("--gamma", type=float, default=training_defaults.gamma)
    parser.add_argument("--tau", type=float, default=training_defaults.tau)
    parser.add_argument("--policy-delay", type=int, default=training_defaults.policy_delay)
    parser.add_argument(
        "--train-frequency-steps",
        type=int,
        default=training_defaults.train_frequency_steps,
    )
    parser.add_argument("--gradient-steps", type=int, default=training_defaults.gradient_steps)
    parser.add_argument("--device", type=str, default=training_defaults.device)
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> Td3TrainingConfig:
    return Td3TrainingConfig(
        total_timesteps=args.maximum_timesteps,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        train_frequency_steps=args.train_frequency_steps,
        gradient_steps=args.gradient_steps,
        device=args.device,
    )


def build_convergence_config(args: argparse.Namespace) -> Td3ConvergenceTrainingConfig:
    return Td3ConvergenceTrainingConfig(
        eval_frequency_steps=args.eval_frequency_steps,
        evaluation_episode_count=args.evaluation_episodes,
        patience_evaluation_count=args.patience_evaluations,
        minimum_training_timesteps=args.minimum_timesteps,
        maximum_training_timesteps=args.maximum_timesteps,
        minimum_reward_improvement_float=args.minimum_reward_improvement,
        evaluation_seed_start=args.evaluation_seed_start,
        sb3_log_interval=args.sb3_log_interval,
        output_directory=args.output_dir,
        best_model_name=args.best_model_name,
        latest_model_name=args.latest_model_name,
    )


def validate_convergence_config(config: Td3ConvergenceTrainingConfig) -> None:
    if config.eval_frequency_steps <= 0:
        raise ValueError("eval_frequency_steps must be positive.")
    if config.evaluation_episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive.")
    if config.patience_evaluation_count <= 0:
        raise ValueError("patience_evaluation_count must be positive.")
    if config.minimum_training_timesteps < 0:
        raise ValueError("minimum_training_timesteps cannot be negative.")
    if config.maximum_training_timesteps <= 0:
        raise ValueError("maximum_training_timesteps must be positive.")
    if config.maximum_training_timesteps < config.minimum_training_timesteps:
        raise ValueError("maximum_training_timesteps must be >= minimum_training_timesteps.")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_evaluation_history(history: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(make_json_safe(history), indent=2))
    if not history:
        csv_path.write_text("")
        return
    fieldnames = sorted({key for row in history for key in row.keys()})
    with csv_path.open("w", newline="") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def evaluate_td3_model(
    model,
    simulation_config: SimulationConfig,
    convergence_config: Td3ConvergenceTrainingConfig,
) -> dict[str, float]:
    evaluator = MonteCarloEvaluator(
        trial_count=convergence_config.evaluation_episode_count,
        seed_start=convergence_config.evaluation_seed_start,
    )

    def environment_factory() -> LeaderSynchronizationEnv:
        return LeaderSynchronizationEnv(simulation_config)

    policy = Td3PolicyWrapper(model, deterministic=True)
    result = evaluator.evaluate_policy("TD3", policy, environment_factory)
    return result.metric_mean_dictionary | {
        f"std_{key}": value for key, value in result.metric_std_dictionary.items()
    }


def print_startup_summary(
    simulation_config: SimulationConfig,
    training_config: Td3TrainingConfig,
    convergence_config: Td3ConvergenceTrainingConfig,
    output_dir: Path,
) -> None:
    print("=" * 80, flush=True)
    print("TD3 convergence training", flush=True)
    print("=" * 80, flush=True)
    print(f"seed: {simulation_config.random_seed}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"eval_frequency_steps: {convergence_config.eval_frequency_steps}", flush=True)
    print(f"evaluation_episodes: {convergence_config.evaluation_episode_count}", flush=True)
    print(f"patience_evaluations: {convergence_config.patience_evaluation_count}", flush=True)
    print(f"minimum_training_timesteps: {convergence_config.minimum_training_timesteps}", flush=True)
    print(f"maximum_training_timesteps: {convergence_config.maximum_training_timesteps}", flush=True)
    print(f"minimum_reward_improvement: {convergence_config.minimum_reward_improvement_float}", flush=True)
    print(f"learning_rate: {training_config.learning_rate}", flush=True)
    print(f"device: {training_config.device}", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    args = parse_arguments()
    simulation_config = SimulationConfig(random_seed=args.seed)
    training_config = build_training_config(args)
    convergence_config = build_convergence_config(args)
    validate_convergence_config(convergence_config)

    output_dir = Path(convergence_config.output_directory)
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print_startup_summary(simulation_config, training_config, convergence_config, output_dir)

    trainer = Td3Trainer(simulation_config, training_config)
    model = trainer.build_model()

    best_mean_reward = float("-inf")
    best_timestep = 0
    evaluations_without_improvement = 0
    trained_timesteps = 0
    evaluation_index = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    best_model_path = models_dir / convergence_config.best_model_name
    latest_model_path = models_dir / convergence_config.latest_model_name
    history_json_path = metrics_dir / "td3_convergence_history.json"
    history_csv_path = metrics_dir / "td3_convergence_history.csv"
    reward_plot_path = plots_dir / "td3_evaluation_reward_convergence.png"
    aoi_plot_path = plots_dir / "td3_evaluation_aoi_convergence.png"

    while trained_timesteps < convergence_config.maximum_training_timesteps:
        next_chunk_timesteps = min(
            convergence_config.eval_frequency_steps,
            convergence_config.maximum_training_timesteps - trained_timesteps,
        )
        print(
            f"\nTraining chunk: start={trained_timesteps}, "
            f"chunk={next_chunk_timesteps}, "
            f"target={trained_timesteps + next_chunk_timesteps}",
            flush=True,
        )
        model.learn(
            total_timesteps=next_chunk_timesteps,
            reset_num_timesteps=False,
            progress_bar=False,
            log_interval=convergence_config.sb3_log_interval,
        )
        trained_timesteps += next_chunk_timesteps
        evaluation_index += 1

        print(
            f"Evaluating at {trained_timesteps} timesteps "
            f"over {convergence_config.evaluation_episode_count} episodes...",
            flush=True,
        )
        metric_mean_dictionary = evaluate_td3_model(model, simulation_config, convergence_config)
        mean_reward = float(metric_mean_dictionary["episode_return_float"])
        reward_std = float(metric_mean_dictionary["std_episode_return_float"])
        mean_aoi = float(metric_mean_dictionary["average_weighted_aoi_float"])
        freshness_violations = float(metric_mean_dictionary["freshness_violation_count_integer"])
        terminal_cpu_violations = float(metric_mean_dictionary["terminal_cpu_violation_count_integer"])
        penalized_score = float(metric_mean_dictionary["penalized_score_float"])
        elapsed_seconds = time.time() - start_time

        improved = mean_reward > best_mean_reward + convergence_config.minimum_reward_improvement_float
        if improved:
            best_mean_reward = mean_reward
            best_timestep = trained_timesteps
            evaluations_without_improvement = 0
            model.save(str(best_model_path))
            improvement_text = "improved"
        else:
            evaluations_without_improvement += 1
            improvement_text = "no improvement"

        model.save(str(latest_model_path))

        history_row = {
            "evaluation_index": evaluation_index,
            "timesteps": trained_timesteps,
            "mean_episode_return_float": mean_reward,
            "std_episode_return_float": reward_std,
            "average_weighted_aoi_float": mean_aoi,
            "freshness_violation_count_integer": freshness_violations,
            "terminal_cpu_violation_count_integer": terminal_cpu_violations,
            "penalized_score_float": penalized_score,
            "best_mean_episode_return_float": best_mean_reward,
            "best_timestep": best_timestep,
            "evaluations_without_improvement": evaluations_without_improvement,
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(history_row)
        save_evaluation_history(history, history_json_path, history_csv_path)
        plot_td3_convergence(
            [row["timesteps"] for row in history],
            [row["mean_episode_return_float"] for row in history],
            output_path=reward_plot_path,
            title="TD3 evaluation reward convergence",
            ylabel="Mean evaluation episode return",
        )
        plot_td3_convergence(
            [row["timesteps"] for row in history],
            [row["average_weighted_aoi_float"] for row in history],
            output_path=aoi_plot_path,
            title="TD3 evaluation weighted AoI during training",
            ylabel="Mean weighted AoI",
        )

        print(
            f"Evaluation {evaluation_index}: "
            f"timesteps={trained_timesteps:,}, "
            f"mean_reward={mean_reward:.3f} ± {reward_std:.3f}, "
            f"mean_aoi={mean_aoi:.3f}, "
            f"freshness_violations={freshness_violations:.3f}, "
            f"terminal_cpu_violations={terminal_cpu_violations:.3f}, "
            f"penalized_score={penalized_score:.3f}, "
            f"best_reward={best_mean_reward:.3f} at {best_timestep:,}, "
            f"patience={evaluations_without_improvement}/"
            f"{convergence_config.patience_evaluation_count}, "
            f"status={improvement_text}, "
            f"elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )
        print(f"Saved latest model: {latest_model_path}.zip", flush=True)
        if improved:
            print(f"Saved new best model: {best_model_path}.zip", flush=True)
        print(f"Saved history: {history_json_path}", flush=True)
        print(f"Saved reward plot: {reward_plot_path}", flush=True)

        reached_minimum_training = trained_timesteps >= convergence_config.minimum_training_timesteps
        exhausted_patience = evaluations_without_improvement >= convergence_config.patience_evaluation_count
        if reached_minimum_training and exhausted_patience:
            print(
                "Stopping early: evaluation reward did not improve for "
                f"{convergence_config.patience_evaluation_count} evaluations "
                f"after the minimum training budget of "
                f"{convergence_config.minimum_training_timesteps:,} timesteps.",
                flush=True,
            )
            break

    report_writer = ReportWriter()
    report = report_writer.build_report(
        config_used=simulation_config,
        seed_used=simulation_config.random_seed,
        model_path=str(best_model_path),
        training_hyperparameters={
            "td3_training_config": asdict(training_config),
            "td3_convergence_training_config": asdict(convergence_config),
            "best_mean_episode_return_float": best_mean_reward,
            "best_timestep": best_timestep,
            "final_timestep": trained_timesteps,
        },
        metrics_json_path=str(history_json_path),
        metrics_csv_path=str(history_csv_path),
        plot_paths=[str(reward_plot_path), str(aoi_plot_path)],
        notes=[
            "Best model is selected by highest mean evaluation episode return.",
            "Evaluation uses the same TD3 policy wrapper as Monte Carlo evaluation.",
        ],
    )
    report_writer.save_report(report, output_dir / "td3_convergence_training_report.json")

    print("\nTraining complete.", flush=True)
    print(f"Best model: {best_model_path}.zip", flush=True)
    print(f"Latest model: {latest_model_path}.zip", flush=True)
    print(f"History JSON: {history_json_path}", flush=True)
    print(f"History CSV: {history_csv_path}", flush=True)
    print(f"Reward plot: {reward_plot_path}", flush=True)
    print(f"AoI plot: {aoi_plot_path}", flush=True)


if __name__ == "__main__":
    main()
