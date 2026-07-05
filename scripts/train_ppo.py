"""Train PPO on the defective RSU zone environment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt.config import PpoTrainingConfig, SimulationConfig  # noqa: E402
from leader_dt.evaluation.reporting import ReportWriter  # noqa: E402
from leader_dt.rl.ppo_agent import PpoTrainer  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    defaults = PpoTrainingConfig()
    parser = argparse.ArgumentParser(description="Train a PPO policy on the leader DT environment.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=defaults.clip_range)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--tensorboard-log-dir", type=str, default=defaults.tensorboard_log_directory)
    parser.add_argument("--monitor-log-dir", type=str, default=defaults.monitor_log_directory)
    parser.add_argument("--checkpoint-frequency-steps", type=int, default=defaults.checkpoint_frequency_steps)
    parser.add_argument("--checkpoint-output-dir", type=str, default=defaults.checkpoint_output_directory)
    parser.add_argument("--output-dir", type=str, default="results/ppo_training")
    parser.add_argument("--model-name", type=str, default="ppo_exact_pair_zone_b")
    parser.add_argument("--device", type=str, default=defaults.device)
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> PpoTrainingConfig:
    return PpoTrainingConfig(
        total_timesteps=args.total_timesteps,
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


def validate_training_config(config: PpoTrainingConfig) -> None:
    if config.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.batch_size > config.n_steps:
        raise ValueError("PPO batch_size should be <= n_steps for a single-environment rollout.")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")
    if config.checkpoint_frequency_steps < 0:
        raise ValueError("checkpoint_frequency_steps cannot be negative.")


def main() -> None:
    args = parse_arguments()
    simulation_config = SimulationConfig(random_seed=args.seed)
    training_config = build_training_config(args)
    validate_training_config(training_config)

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trainer = PpoTrainer(simulation_config, training_config)
    model = trainer.train()
    model_path = models_dir / args.model_name
    trainer.save_model(model, str(model_path))

    report_writer = ReportWriter()
    report = report_writer.build_report(
        config_used=simulation_config,
        seed_used=simulation_config.random_seed,
        model_path=str(model_path),
        training_hyperparameters={"ppo_training_config": asdict(training_config)},
        notes=["Single-shot PPO training run."],
    )
    report_writer.save_report(report, output_dir / "ppo_training_report.json")
    print(f"Saved PPO model: {model_path}.zip", flush=True)
    print(f"Saved report: {output_dir / 'ppo_training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
