"""Train PPO with periodic evaluation and convergence-based early stopping.

This script mirrors the TD3 convergence workflow while keeping PPO-specific
hyperparameters in ``PpoTrainingConfig``. It trains in chunks, evaluates the PPO
policy every configured number of timesteps, saves best/latest models, records
history, and generates convergence plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.config import (  # noqa: E402
    PpoConvergenceTrainingConfig,
    PpoTrainingConfig,
    SimulationConfig,
)
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator  # noqa: E402
from leader_dt.evaluation.reporting import ReportWriter  # noqa: E402
from leader_dt.plotting.convergence_plots import plot_td3_convergence  # noqa: E402
from leader_dt.rl.ppo_agent import PpoTrainer  # noqa: E402
from leader_dt.rl.wrappers import PpoPolicyWrapper  # noqa: E402
from leader_dt.simulator.environment import LeaderSynchronizationEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    convergence_defaults = PpoConvergenceTrainingConfig()
    training_defaults = PpoTrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train PPO until evaluation reward converges, while saving the "
            "best model, latest model, evaluation history, and convergence plots."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-frequency-steps", type=int, default=convergence_defaults.eval_frequency_steps)
    parser.add_argument("--evaluation-episodes", type=int, default=convergence_defaults.evaluation_episode_count)
    parser.add_argument("--patience-evaluations", type=int, default=convergence_defaults.patience_evaluation_count)
    parser.add_argument("--minimum-timesteps", type=int, default=convergence_defaults.minimum_training_timesteps)
    parser.add_argument("--maximum-timesteps", type=int, default=convergence_defaults.maximum_training_timesteps)
    parser.add_argument(
        "--minimum-reward-improvement",
        type=float,
        default=convergence_defaults.minimum_reward_improvement_float,
    )
    parser.add_argument("--evaluation-seed-start", type=int, default=convergence_defaults.evaluation_seed_start)
    parser.add_argument("--output-dir", type=str, default=convergence_defaults.output_directory)
    parser.add_argument("--best-model-name", type=str, default=convergence_defaults.best_model_name)
    parser.add_argument("--latest-model-name", type=str, default=convergence_defaults.latest_model_name)
    parser.add_argument("--sb3-log-interval", type=int, default=convergence_defaults.sb3_log_interval)
    parser.add_argument("--learning-rate", type=float, default=training_defaults.learning_rate)
    parser.add_argument("--n-steps", type=int, default=training_defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument("--n-epochs", type=int, default=training_defaults.n_epochs)
    parser.add_argument("--gamma", type=float, default=training_defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=training_defaults.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=training_defaults.clip_range)
    parser.add_argument("--ent-coef", type=float, default=training_defaults.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=training_defaults.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=training_defaults.max_grad_norm)
    parser.add_argument("--tensorboard-log-dir", type=str, default=training_defaults.tensorboard_log_directory)
    parser.add_argument("--monitor-log-dir", type=str, default=training_defaults.monitor_log_directory)
    parser.add_argument(
        "--checkpoint-frequency-steps",
        type=int,
        default=convergence_defaults.checkpoint_frequency_steps,
    )
    parser.add_argument("--checkpoint-output-dir", type=str, default=training_defaults.checkpoint_output_directory)
    parser.add_argument("--device", type=str, default=training_defaults.device)
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> PpoTrainingConfig:
    return PpoTrainingConfig(
        total_timesteps=args.maximum_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log_directory=args.tensorboard_log_dir,
        monitor_log_directory=args.monitor_log_dir,
        checkpoint_frequency_steps=args.checkpoint_frequency_steps,
        checkpoint_output_directory=args.checkpoint_output_dir,
        device=args.device,
    )


def build_convergence_config(args: argparse.Namespace) -> PpoConvergenceTrainingConfig:
    return PpoConvergenceTrainingConfig(
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
        checkpoint_frequency_steps=args.checkpoint_frequency_steps,
    )


def validate_convergence_config(
    training_config: PpoTrainingConfig,
    convergence_config: PpoConvergenceTrainingConfig,
) -> None:
    if convergence_config.eval_frequency_steps <= 0:
        raise ValueError("eval_frequency_steps must be positive.")
    if convergence_config.evaluation_episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive.")
    if convergence_config.patience_evaluation_count <= 0:
        raise ValueError("patience_evaluation_count must be positive.")
    if convergence_config.minimum_training_timesteps < 0:
        raise ValueError("minimum_training_timesteps cannot be negative.")
    if convergence_config.maximum_training_timesteps <= 0:
        raise ValueError("maximum_training_timesteps must be positive.")
    if convergence_config.maximum_training_timesteps < convergence_config.minimum_training_timesteps:
        raise ValueError("maximum_training_timesteps must be >= minimum_training_timesteps.")
    if convergence_config.checkpoint_frequency_steps < 0:
        raise ValueError("checkpoint_frequency_steps cannot be negative.")
    if training_config.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if training_config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if training_config.batch_size > training_config.n_steps:
        raise ValueError("PPO batch_size should be <= n_steps for a single-environment rollout.")
    if training_config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")


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


def evaluate_ppo_model(
    model,
    simulation_config: SimulationConfig,
    convergence_config: PpoConvergenceTrainingConfig,
) -> dict[str, float]:
    evaluator = MonteCarloEvaluator(
        trial_count=convergence_config.evaluation_episode_count,
        seed_start=convergence_config.evaluation_seed_start,
    )

    def environment_factory():
        return Monitor(LeaderSynchronizationEnv(simulation_config), filename=None)

    policy = PpoPolicyWrapper(model, deterministic=True)
    result = evaluator.evaluate_policy("PPO", policy, environment_factory)
    return result.metric_mean_dictionary | {
        f"std_{key}": value for key, value in result.metric_std_dictionary.items()
    }


def print_startup_summary(
    simulation_config: SimulationConfig,
    training_config: PpoTrainingConfig,
    convergence_config: PpoConvergenceTrainingConfig,
    output_dir: Path,
) -> None:
    print("=" * 80, flush=True)
    print("PPO convergence training", flush=True)
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
    print(f"n_steps: {training_config.n_steps}", flush=True)
    print(f"batch_size: {training_config.batch_size}", flush=True)
    print(f"n_epochs: {training_config.n_epochs}", flush=True)
    print(f"gamma: {training_config.gamma}", flush=True)
    print(f"gae_lambda: {training_config.gae_lambda}", flush=True)
    print(f"clip_range: {training_config.clip_range}", flush=True)
    print(f"ent_coef: {training_config.ent_coef}", flush=True)
    print(f"vf_coef: {training_config.vf_coef}", flush=True)
    print(f"max_grad_norm: {training_config.max_grad_norm}", flush=True)
    print(f"tensorboard_log_directory: {training_config.tensorboard_log_directory}", flush=True)
    print(f"monitor_log_directory: {training_config.monitor_log_directory}", flush=True)
    print(f"checkpoint_frequency_steps: {convergence_config.checkpoint_frequency_steps}", flush=True)
    print(f"device: {training_config.device}", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    args = parse_arguments()
    simulation_config = SimulationConfig(random_seed=args.seed)
    training_config = build_training_config(args)
    convergence_config = build_convergence_config(args)
    validate_convergence_config(training_config, convergence_config)

    output_dir = Path(convergence_config.output_directory)
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    checkpoints_dir = output_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print_startup_summary(simulation_config, training_config, convergence_config, output_dir)

    trainer = PpoTrainer(simulation_config, training_config)
    model = trainer.build_model()

    best_mean_reward = float("-inf")
    best_timestep = 0
    evaluations_without_improvement = 0
    trained_timesteps = 0
    evaluation_index = 0
    history: list[dict[str, Any]] = []
    saved_checkpoint_timesteps: set[int] = set()
    start_time = time.time()

    best_model_path = models_dir / convergence_config.best_model_name
    latest_model_path = models_dir / convergence_config.latest_model_name
    history_json_path = metrics_dir / "ppo_convergence_history.json"
    history_csv_path = metrics_dir / "ppo_convergence_history.csv"
    reward_plot_path = plots_dir / "ppo_evaluation_reward_convergence.png"
    aoi_plot_path = plots_dir / "ppo_evaluation_aoi_convergence.png"

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
        metric_mean_dictionary = evaluate_ppo_model(model, simulation_config, convergence_config)
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
        checkpoint_saved_path: str | None = None
        if (
            convergence_config.checkpoint_frequency_steps > 0
            and trained_timesteps % convergence_config.checkpoint_frequency_steps == 0
            and trained_timesteps not in saved_checkpoint_timesteps
        ):
            checkpoint_path = checkpoints_dir / f"ppo_checkpoint_{trained_timesteps}_steps"
            model.save(str(checkpoint_path))
            checkpoint_saved_path = f"{checkpoint_path}.zip"
            saved_checkpoint_timesteps.add(trained_timesteps)

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
            "checkpoint_saved_path": checkpoint_saved_path,
        }
        history.append(history_row)
        save_evaluation_history(history, history_json_path, history_csv_path)
        plot_td3_convergence(
            [row["timesteps"] for row in history],
            [row["mean_episode_return_float"] for row in history],
            output_path=reward_plot_path,
            title="PPO evaluation reward convergence",
            ylabel="Mean evaluation episode return",
        )
        plot_td3_convergence(
            [row["timesteps"] for row in history],
            [row["average_weighted_aoi_float"] for row in history],
            output_path=aoi_plot_path,
            title="PPO evaluation weighted AoI during training",
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
        if checkpoint_saved_path is not None:
            print(f"Saved periodic checkpoint: {checkpoint_saved_path}", flush=True)
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
            "ppo_training_config": asdict(training_config),
            "ppo_convergence_training_config": asdict(convergence_config),
            "best_mean_episode_return_float": best_mean_reward,
            "best_timestep": best_timestep,
            "final_timestep": trained_timesteps,
        },
        metrics_json_path=str(history_json_path),
        metrics_csv_path=str(history_csv_path),
        plot_paths=[str(reward_plot_path), str(aoi_plot_path)],
        notes=[
            "Best model is selected by highest mean evaluation episode return.",
            "Evaluation uses deterministic PPO model prediction through PpoPolicyWrapper.",
        ],
    )
    report_writer.save_report(report, output_dir / "ppo_convergence_training_report.json")

    print("\nTraining complete.", flush=True)
    print(f"Best model: {best_model_path}.zip", flush=True)
    print(f"Latest model: {latest_model_path}.zip", flush=True)
    print(f"History JSON: {history_json_path}", flush=True)
    print(f"History CSV: {history_csv_path}", flush=True)
    print(f"Reward plot: {reward_plot_path}", flush=True)
    print(f"AoI plot: {aoi_plot_path}", flush=True)
    print(f"Periodic checkpoints directory: {checkpoints_dir}", flush=True)


if __name__ == "__main__":
    main()
