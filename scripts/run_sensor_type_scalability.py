"""Train-per-point scalability experiment for the number of sensor types.

This script is intentionally separate from ``run_sensitivity.py`` because the
number of active sensor types changes the effective decision structure.  For
that parameter, TD3 and PPO are trained from scratch for each sensor-type-count
value and each training seed before the point is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leader_dt import constants  # noqa: E402
from leader_dt.config import SimulationConfig  # noqa: E402
from leader_dt.evaluation.monte_carlo import MonteCarloEvaluator, MonteCarloResult  # noqa: E402
from leader_dt.evaluation.policy_factory import build_policy_dictionary  # noqa: E402
from leader_dt.evaluation.reporting import ReportWriter  # noqa: E402
from leader_dt.evaluation.sensitivity import SensitivityPointResult  # noqa: E402
from leader_dt.plotting.sensitivity_plots import plot_sensitivity_curves  # noqa: E402
from leader_dt.rl.wrappers import PpoPolicyWrapper, Td3PolicyWrapper  # noqa: E402
from leader_dt.simulator.environment import LeaderSynchronizationEnv  # noqa: E402


DEFAULT_SENSOR_TYPE_VALUES = "4,5,6,7,8,10,12,14,15,16"
DEFAULT_TRAINING_SEEDS = "1,2,3"
DEFAULT_MAXIMUM_TIMESTEPS = 3_000_000
DEFAULT_MINIMUM_TIMESTEPS = 3_000_000
DEFAULT_EVAL_FREQUENCY_STEPS = 100_000
DEFAULT_TRAINING_EVALUATION_EPISODES = 50
DEFAULT_PATIENCE_EVALUATIONS = 999_999
DEFAULT_MINIMUM_REWARD_IMPROVEMENT = 50.0
DEFAULT_EVALUATION_TRIALS = 500
DEFAULT_EVALUATION_SEED_START = 50_000


MODEL_FILENAME_BY_ALGORITHM = {
    "td3": "best_td3_exact_pair_zone_b.zip",
    "ppo": "best_ppo_exact_pair_zone_b.zip",
}


BEST_MODEL_STEM_BY_ALGORITHM = {
    "td3": "best_td3_exact_pair_zone_b",
    "ppo": "best_ppo_exact_pair_zone_b",
}


LATEST_MODEL_STEM_BY_ALGORITHM = {
    "td3": "latest_td3_exact_pair_zone_b",
    "ppo": "latest_ppo_exact_pair_zone_b",
}


def parse_int_list(raw_values: str) -> list[int]:
    """Parse comma-separated integers while preserving input order."""
    parsed_values: list[int] = []
    for item in raw_values.split(","):
        stripped_item = item.strip()
        if stripped_item == "":
            continue
        parsed_values.append(int(stripped_item))
    if not parsed_values:
        raise ValueError("At least one integer value is required.")
    return parsed_values


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the scalability experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Run sensor-type-count scalability by retraining TD3 and PPO from "
            "scratch for each sensor-type-count value and training seed."
        )
    )
    parser.add_argument("--sensor-type-values", type=str, default=DEFAULT_SENSOR_TYPE_VALUES)
    parser.add_argument("--training-seeds", type=str, default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--trials", type=int, default=DEFAULT_EVALUATION_TRIALS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_EVALUATION_SEED_START)
    parser.add_argument("--output-dir", type=str, default="results/sensor_type_scalability")

    parser.add_argument("--maximum-timesteps", type=int, default=DEFAULT_MAXIMUM_TIMESTEPS)
    parser.add_argument("--minimum-timesteps", type=int, default=DEFAULT_MINIMUM_TIMESTEPS)
    parser.add_argument("--eval-frequency-steps", type=int, default=DEFAULT_EVAL_FREQUENCY_STEPS)
    parser.add_argument("--training-evaluation-episodes", type=int, default=DEFAULT_TRAINING_EVALUATION_EPISODES)
    parser.add_argument("--patience-evaluations", type=int, default=DEFAULT_PATIENCE_EVALUATIONS)
    parser.add_argument("--minimum-reward-improvement", type=float, default=DEFAULT_MINIMUM_REWARD_IMPROVEMENT)
    parser.add_argument("--checkpoint-frequency-steps", type=int, default=250_000)
    parser.add_argument("--evaluation-seed-start", type=int, default=10_000)

    parser.add_argument("--td3-device", type=str, default=constants.DEFAULT_DEVICE)
    parser.add_argument("--ppo-device", type=str, default="cpu")
    parser.add_argument("--evaluation-device", type=str, default="cpu")

    parser.add_argument("--td3-learning-rate", type=float, default=constants.DEFAULT_LEARNING_RATE)
    parser.add_argument("--td3-learning-starts", type=int, default=constants.DEFAULT_LEARNING_STARTS)
    parser.add_argument("--td3-buffer-size", type=int, default=constants.DEFAULT_BUFFER_SIZE)
    parser.add_argument("--td3-batch-size", type=int, default=constants.DEFAULT_BATCH_SIZE)
    parser.add_argument("--td3-action-noise-sigma", type=float, default=0.05)
    parser.add_argument("--td3-target-policy-noise", type=float, default=constants.DEFAULT_TARGET_POLICY_NOISE)
    parser.add_argument("--td3-target-noise-clip", type=float, default=constants.DEFAULT_TARGET_NOISE_CLIP)

    parser.add_argument("--ppo-learning-rate", type=float, default=constants.DEFAULT_PPO_LEARNING_RATE)
    parser.add_argument("--ppo-n-steps", type=int, default=constants.DEFAULT_PPO_N_STEPS)
    parser.add_argument("--ppo-batch-size", type=int, default=constants.DEFAULT_PPO_BATCH_SIZE)
    parser.add_argument("--ppo-n-epochs", type=int, default=constants.DEFAULT_PPO_N_EPOCHS)
    parser.add_argument("--ppo-clip-range", type=float, default=constants.DEFAULT_PPO_CLIP_RANGE)
    parser.add_argument("--ppo-ent-coef", type=float, default=constants.DEFAULT_PPO_ENT_COEF)
    parser.add_argument("--ppo-vf-coef", type=float, default=constants.DEFAULT_PPO_VF_COEF)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=constants.DEFAULT_PPO_MAX_GRAD_NORM)

    parser.add_argument("--greedy-lambda-cpu", type=float, default=constants.DEFAULT_GREEDY_CPU_LAMBDA)
    parser.add_argument(
        "--greedy-requested-accuracy-fraction",
        type=float,
        default=constants.DEFAULT_GREEDY_REQUESTED_ACCURACY_FRACTION,
    )
    return parser.parse_args()


def build_simulation_config(sensor_type_count: int, random_seed: int | None = None) -> SimulationConfig:
    """Build a simulation config with the requested number of sensor types."""
    base_config = SimulationConfig(random_seed=random_seed)
    return replace(
        base_config,
        system=replace(
            base_config.system,
            sensor_type_count=int(sensor_type_count),
        ),
    )


def build_expected_manifest(
    *,
    algorithm: str,
    sensor_type_count: int,
    training_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return metadata that uniquely identifies a reusable trained checkpoint."""
    simulation_config = build_simulation_config(
        sensor_type_count=sensor_type_count,
        random_seed=training_seed,
    )
    common_manifest: dict[str, Any] = {
        "algorithm": algorithm,
        "sensor_type_count": int(sensor_type_count),
        "training_seed": int(training_seed),
        "simulation_config": asdict(simulation_config),
        "maximum_timesteps": int(args.maximum_timesteps),
        "minimum_timesteps": int(args.minimum_timesteps),
        "eval_frequency_steps": int(args.eval_frequency_steps),
        "training_evaluation_episodes": int(args.training_evaluation_episodes),
        "patience_evaluations": int(args.patience_evaluations),
        "minimum_reward_improvement": float(args.minimum_reward_improvement),
        "checkpoint_frequency_steps": int(args.checkpoint_frequency_steps),
        "evaluation_seed_start": int(args.evaluation_seed_start),
        "active_policy_set": ["Greedy", "Proximity Greedy", "TD3", "PPO"],
        "max_supported_sensor_types": len(constants.DEFAULT_SENSOR_DEFINITIONS),
        "max_pair_count_for_action_space": int(simulation_config.system.max_pair_count_for_action_space),
    }

    if algorithm == "td3":
        common_manifest["algorithm_hyperparameters"] = {
            "device": args.td3_device,
            "learning_rate": float(args.td3_learning_rate),
            "learning_starts": int(args.td3_learning_starts),
            "buffer_size": int(args.td3_buffer_size),
            "batch_size": int(args.td3_batch_size),
            "action_noise_sigma": float(args.td3_action_noise_sigma),
            "target_policy_noise": float(args.td3_target_policy_noise),
            "target_noise_clip": float(args.td3_target_noise_clip),
        }
    elif algorithm == "ppo":
        ppo_n_steps = max(1, min(int(args.ppo_n_steps), int(args.maximum_timesteps)))
        ppo_batch_size = max(1, min(int(args.ppo_batch_size), ppo_n_steps))
        common_manifest["algorithm_hyperparameters"] = {
            "device": args.ppo_device,
            "learning_rate": float(args.ppo_learning_rate),
            "n_steps": int(ppo_n_steps),
            "batch_size": int(ppo_batch_size),
            "n_epochs": int(args.ppo_n_epochs),
            "clip_range": float(args.ppo_clip_range),
            "ent_coef": float(args.ppo_ent_coef),
            "vf_coef": float(args.ppo_vf_coef),
            "max_grad_norm": float(args.ppo_max_grad_norm),
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return common_manifest


def manifest_matches(manifest_path: Path, expected_manifest: dict[str, Any]) -> bool:
    """Return True only when a manifest exists and all expected fields match."""
    if not manifest_path.exists():
        return False
    try:
        actual_manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return all(actual_manifest.get(key) == value for key, value in expected_manifest.items())


def save_manifest(manifest_path: Path, expected_manifest: dict[str, Any], model_path: Path) -> None:
    """Save checkpoint metadata after successful training."""
    manifest = {
        **expected_manifest,
        "model_path": str(model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def model_path_for_run(output_dir: Path, algorithm: str) -> Path:
    """Return the best model path created by the convergence training script."""
    return output_dir / "models" / MODEL_FILENAME_BY_ALGORITHM[algorithm]


def training_output_dir(base_output_dir: Path, sensor_type_count: int, training_seed: int, algorithm: str) -> Path:
    """Return the per-point, per-seed, per-algorithm training directory."""
    return (
        base_output_dir
        / f"sensor_type_{int(sensor_type_count)}"
        / f"seed_{int(training_seed)}"
        / algorithm
    )


def build_training_command(
    *,
    algorithm: str,
    sensor_type_count: int,
    training_seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build the subprocess command that invokes the existing training script."""
    if algorithm == "td3":
        script_path = PROJECT_ROOT / "scripts" / "train_td3_until_convergence.py"
        command = [
            sys.executable,
            str(script_path),
            "--seed",
            str(training_seed),
            "--sensor-type-count",
            str(sensor_type_count),
            "--maximum-timesteps",
            str(args.maximum_timesteps),
            "--minimum-timesteps",
            str(args.minimum_timesteps),
            "--eval-frequency-steps",
            str(args.eval_frequency_steps),
            "--evaluation-episodes",
            str(args.training_evaluation_episodes),
            "--patience-evaluations",
            str(args.patience_evaluations),
            "--minimum-reward-improvement",
            str(args.minimum_reward_improvement),
            "--evaluation-seed-start",
            str(args.evaluation_seed_start),
            "--checkpoint-frequency-steps",
            str(args.checkpoint_frequency_steps),
            "--output-dir",
            str(output_dir),
            "--tensorboard-log-dir",
            str(output_dir / "tensorboard"),
            "--monitor-log-dir",
            str(output_dir / "monitor"),
            "--checkpoint-output-dir",
            str(output_dir / "checkpoints"),
            "--device",
            str(args.td3_device),
            "--learning-rate",
            str(args.td3_learning_rate),
            "--learning-starts",
            str(args.td3_learning_starts),
            "--buffer-size",
            str(args.td3_buffer_size),
            "--batch-size",
            str(args.td3_batch_size),
            "--action-noise-sigma",
            str(args.td3_action_noise_sigma),
            "--target-policy-noise",
            str(args.td3_target_policy_noise),
            "--target-noise-clip",
            str(args.td3_target_noise_clip),
            "--best-model-name",
            BEST_MODEL_STEM_BY_ALGORITHM[algorithm],
            "--latest-model-name",
            LATEST_MODEL_STEM_BY_ALGORITHM[algorithm],
        ]
        return command

    if algorithm == "ppo":
        script_path = PROJECT_ROOT / "scripts" / "train_ppo_until_convergence.py"
        ppo_n_steps = max(1, min(int(args.ppo_n_steps), int(args.maximum_timesteps)))
        ppo_batch_size = max(1, min(int(args.ppo_batch_size), ppo_n_steps))
        command = [
            sys.executable,
            str(script_path),
            "--seed",
            str(training_seed),
            "--sensor-type-count",
            str(sensor_type_count),
            "--maximum-timesteps",
            str(args.maximum_timesteps),
            "--minimum-timesteps",
            str(args.minimum_timesteps),
            "--eval-frequency-steps",
            str(args.eval_frequency_steps),
            "--evaluation-episodes",
            str(args.training_evaluation_episodes),
            "--patience-evaluations",
            str(args.patience_evaluations),
            "--minimum-reward-improvement",
            str(args.minimum_reward_improvement),
            "--evaluation-seed-start",
            str(args.evaluation_seed_start),
            "--checkpoint-frequency-steps",
            str(args.checkpoint_frequency_steps),
            "--output-dir",
            str(output_dir),
            "--tensorboard-log-dir",
            str(output_dir / "tensorboard"),
            "--monitor-log-dir",
            str(output_dir / "monitor"),
            "--checkpoint-output-dir",
            str(output_dir / "checkpoints"),
            "--device",
            str(args.ppo_device),
            "--learning-rate",
            str(args.ppo_learning_rate),
            "--n-steps",
            str(ppo_n_steps),
            "--batch-size",
            str(ppo_batch_size),
            "--n-epochs",
            str(args.ppo_n_epochs),
            "--clip-range",
            str(args.ppo_clip_range),
            "--ent-coef",
            str(args.ppo_ent_coef),
            "--vf-coef",
            str(args.ppo_vf_coef),
            "--max-grad-norm",
            str(args.ppo_max_grad_norm),
            "--best-model-name",
            BEST_MODEL_STEM_BY_ALGORITHM[algorithm],
            "--latest-model-name",
            LATEST_MODEL_STEM_BY_ALGORITHM[algorithm],
        ]
        return command

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def ensure_trained_model(
    *,
    algorithm: str,
    sensor_type_count: int,
    training_seed: int,
    base_output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Train a model unless a matching checkpoint and manifest already exist."""
    output_dir = training_output_dir(base_output_dir, sensor_type_count, training_seed, algorithm)
    model_path = model_path_for_run(output_dir, algorithm)
    manifest_path = output_dir / "training_manifest.json"
    expected_manifest = build_expected_manifest(
        algorithm=algorithm,
        sensor_type_count=sensor_type_count,
        training_seed=training_seed,
        args=args,
    )

    if model_path.exists() and manifest_matches(manifest_path, expected_manifest):
        print(
            f"[skip] {algorithm.upper()} sensor_type_count={sensor_type_count} "
            f"seed={training_seed}: matching checkpoint found at {model_path}",
            flush=True,
        )
        return model_path

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_training_command(
        algorithm=algorithm,
        sensor_type_count=sensor_type_count,
        training_seed=training_seed,
        output_dir=output_dir,
        args=args,
    )
    log_path = output_dir / "training_stdout.log"
    print(
        f"[train] {algorithm.upper()} sensor_type_count={sensor_type_count} "
        f"seed={training_seed}",
        flush=True,
    )
    print(" ".join(command), flush=True)
    start_time = time.time()
    with log_path.open("w") as log_file:
        completed_process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_seconds = time.time() - start_time

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"{algorithm.upper()} training failed for sensor_type_count={sensor_type_count}, "
            f"seed={training_seed}. See log: {log_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"{algorithm.upper()} training completed but expected model was not found: {model_path}"
        )

    save_manifest(
        manifest_path,
        {
            **expected_manifest,
            "training_elapsed_seconds": elapsed_seconds,
            "training_stdout_log": str(log_path),
        },
        model_path,
    )
    print(
        f"[done] {algorithm.upper()} sensor_type_count={sensor_type_count} "
        f"seed={training_seed}: {elapsed_seconds:.1f}s",
        flush=True,
    )
    return model_path


def make_environment_factory(simulation_config: SimulationConfig):
    """Create a raw environment factory for Monte Carlo evaluation."""

    def environment_factory() -> LeaderSynchronizationEnv:
        return LeaderSynchronizationEnv(simulation_config)

    return environment_factory


def aggregate_monte_carlo_results(
    policy_name: str,
    result_list: list[MonteCarloResult],
    training_seeds: list[int] | None = None,
) -> MonteCarloResult:
    """Aggregate several Monte Carlo results into one paper-facing result."""
    if not result_list:
        raise ValueError("result_list must not be empty.")

    flattened_rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(result_list):
        training_seed = None if training_seeds is None else training_seeds[result_index]
        for row in result.per_trial_metric_list:
            enriched_row = dict(row)
            if training_seed is not None:
                enriched_row["training_seed_integer"] = int(training_seed)
            flattened_rows.append(enriched_row)

    metric_keys = [
        key
        for key in flattened_rows[0].keys()
        if key != "training_seed_integer" and isinstance(flattened_rows[0][key], (int, float, np.integer, np.floating))
    ]
    mean = {
        key: float(np.mean([float(row[key]) for row in flattened_rows]))
        for key in metric_keys
    }
    std = {
        key: float(np.std([float(row[key]) for row in flattened_rows]))
        for key in metric_keys
    }
    return MonteCarloResult(
        policy_name=policy_name,
        metric_mean_dictionary=mean,
        metric_std_dictionary=std,
        per_trial_metric_list=flattened_rows,
    )


def evaluate_baseline_policies(
    *,
    simulation_config: SimulationConfig,
    args: argparse.Namespace,
) -> dict[str, MonteCarloResult]:
    """Evaluate Greedy and Proximity Greedy once for one sensor-type count."""
    evaluator = MonteCarloEvaluator(trial_count=args.trials, seed_start=args.seed_start)
    return evaluator.evaluate_policy_dictionary(
        build_policy_dictionary(
            greedy_lambda_cpu=args.greedy_lambda_cpu,
            greedy_requested_accuracy_fraction=args.greedy_requested_accuracy_fraction,
        ),
        make_environment_factory(simulation_config),
    )


def evaluate_td3_models(
    *,
    model_paths: list[Path],
    training_seeds: list[int],
    simulation_config: SimulationConfig,
    args: argparse.Namespace,
) -> MonteCarloResult:
    """Evaluate and aggregate TD3 models trained from different seeds."""
    from stable_baselines3 import TD3

    evaluator = MonteCarloEvaluator(trial_count=args.trials, seed_start=args.seed_start)
    seed_results: list[MonteCarloResult] = []
    for model_path, training_seed in zip(model_paths, training_seeds, strict=True):
        model = TD3.load(str(model_path), device=args.evaluation_device)
        policy = Td3PolicyWrapper(model, deterministic=True)
        result = evaluator.evaluate_policy(
            f"TD3 seed {training_seed}",
            policy,
            make_environment_factory(simulation_config),
        )
        seed_results.append(result)
    return aggregate_monte_carlo_results("TD3", seed_results, training_seeds)


def evaluate_ppo_models(
    *,
    model_paths: list[Path],
    training_seeds: list[int],
    simulation_config: SimulationConfig,
    args: argparse.Namespace,
) -> MonteCarloResult:
    """Evaluate and aggregate PPO models trained from different seeds."""
    from stable_baselines3 import PPO

    evaluator = MonteCarloEvaluator(trial_count=args.trials, seed_start=args.seed_start)
    seed_results: list[MonteCarloResult] = []
    for model_path, training_seed in zip(model_paths, training_seeds, strict=True):
        model = PPO.load(str(model_path), device=args.evaluation_device)
        policy = PpoPolicyWrapper(model, deterministic=True)
        result = evaluator.evaluate_policy(
            f"PPO seed {training_seed}",
            policy,
            make_environment_factory(simulation_config),
        )
        seed_results.append(result)
    return aggregate_monte_carlo_results("PPO", seed_results, training_seeds)


def build_csv_rows(sensitivity_results: list[SensitivityPointResult]) -> list[dict[str, Any]]:
    """Convert scalability results to compact mean/std CSV rows."""
    rows: list[dict[str, Any]] = []
    for point in sensitivity_results:
        for policy_name, result in point.policy_results.items():
            mean_row = {
                "parameter_name": point.parameter_name,
                "parameter_value": point.parameter_value,
                "policy_name": policy_name,
                "statistic": "mean",
            }
            mean_row.update(result.metric_mean_dictionary)
            rows.append(mean_row)

            std_row = {
                "parameter_name": point.parameter_name,
                "parameter_value": point.parameter_value,
                "policy_name": policy_name,
                "statistic": "std",
            }
            std_row.update(result.metric_std_dictionary)
            rows.append(std_row)
    return rows


def save_per_trial_csv(sensitivity_results: list[SensitivityPointResult], output_path: Path) -> Path:
    """Save all per-trial rows, including RL training seed identifiers."""
    rows: list[dict[str, Any]] = []
    for point in sensitivity_results:
        for policy_name, result in point.policy_results.items():
            for trial_index, metrics in enumerate(result.per_trial_metric_list):
                row = {
                    "parameter_name": point.parameter_name,
                    "parameter_value": point.parameter_value,
                    "policy_name": policy_name,
                    "trial_index": trial_index,
                }
                row.update(metrics)
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return output_path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def run_scalability_experiment(args: argparse.Namespace) -> list[SensitivityPointResult]:
    """Run the train-per-point sensor-type scalability experiment."""
    sensor_type_values = parse_int_list(args.sensor_type_values)
    training_seeds = parse_int_list(args.training_seeds)
    max_supported_sensor_types = len(constants.DEFAULT_SENSOR_DEFINITIONS)
    for value in sensor_type_values:
        if value <= 0:
            raise ValueError("sensor-type values must be positive.")
        if value > max_supported_sensor_types:
            raise ValueError(
                f"sensor_type_count={value} exceeds configured sensor definitions "
                f"({max_supported_sensor_types})."
            )

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_results: list[SensitivityPointResult] = []

    print("Sensor-type scalability experiment", flush=True)
    print("sensor_type_values:", sensor_type_values, flush=True)
    print("training_seeds:", training_seeds, flush=True)
    print("trials per policy/model:", args.trials, flush=True)
    print("output_dir:", base_output_dir, flush=True)

    for sensor_type_count in sensor_type_values:
        print("=" * 80, flush=True)
        print(f"sensor_type_count={sensor_type_count}", flush=True)
        print("=" * 80, flush=True)

        td3_model_paths: list[Path] = []
        ppo_model_paths: list[Path] = []
        for training_seed in training_seeds:
            td3_model_paths.append(
                ensure_trained_model(
                    algorithm="td3",
                    sensor_type_count=sensor_type_count,
                    training_seed=training_seed,
                    base_output_dir=base_output_dir,
                    args=args,
                )
            )
            ppo_model_paths.append(
                ensure_trained_model(
                    algorithm="ppo",
                    sensor_type_count=sensor_type_count,
                    training_seed=training_seed,
                    base_output_dir=base_output_dir,
                    args=args,
                )
            )

        simulation_config = build_simulation_config(sensor_type_count, random_seed=args.seed_start)
        policy_results = evaluate_baseline_policies(simulation_config=simulation_config, args=args)
        policy_results["TD3"] = evaluate_td3_models(
            model_paths=td3_model_paths,
            training_seeds=training_seeds,
            simulation_config=simulation_config,
            args=args,
        )
        policy_results["PPO"] = evaluate_ppo_models(
            model_paths=ppo_model_paths,
            training_seeds=training_seeds,
            simulation_config=simulation_config,
            args=args,
        )

        sensitivity_results.append(
            SensitivityPointResult(
                parameter_name="sensor_type_count",
                parameter_value=sensor_type_count,
                policy_results=policy_results,
            )
        )
    return sensitivity_results


def save_outputs(args: argparse.Namespace, sensitivity_results: list[SensitivityPointResult]) -> None:
    """Save JSON, CSV, plot, and high-level report outputs."""
    output_dir = Path(args.output_dir)
    report_writer = ReportWriter()

    metrics_json_path = output_dir / "sensor_type_scalability.json"
    metrics_csv_path = output_dir / "sensor_type_scalability_summary.csv"
    per_trial_csv_path = output_dir / "sensor_type_scalability_per_trial.csv"
    plot_path = output_dir / "plots" / "sensor_type_scalability.png"

    payload = {
        "experiment_name": "sensor_type_scalability_train_per_point",
        "parameter_name": "sensor_type_count",
        "sensor_type_values": parse_int_list(args.sensor_type_values),
        "training_seeds": parse_int_list(args.training_seeds),
        "trials": args.trials,
        "seed_start": args.seed_start,
        "maximum_timesteps": args.maximum_timesteps,
        "minimum_timesteps": args.minimum_timesteps,
        "eval_frequency_steps": args.eval_frequency_steps,
        "training_evaluation_episodes": args.training_evaluation_episodes,
        "patience_evaluations": args.patience_evaluations,
        "minimum_reward_improvement": args.minimum_reward_improvement,
        "notes": [
            "TD3 and PPO are retrained from scratch for each sensor_type_count value.",
            "Each TD3/PPO point aggregates over all configured training seeds.",
            "Greedy and Proximity Greedy are deterministic baselines and are evaluated without training.",
            "Plot uses thesis style and shows means only; standard deviations remain in JSON/CSV.",
        ],
        "points": sensitivity_results,
    }
    report_writer.save_metrics_json(payload, metrics_json_path)
    report_writer.save_metrics_csv(build_csv_rows(sensitivity_results), metrics_csv_path)
    save_per_trial_csv(sensitivity_results, per_trial_csv_path)
    plot_sensitivity_curves(
        sensitivity_results,
        output_path=plot_path,
        title="Scalability: sensor type count",
        xlabel="Number of sensor types",
        ylabel="Average weighted AoI",
    )

    report = report_writer.build_report(
        config_used=build_simulation_config(sensor_type_count=parse_int_list(args.sensor_type_values)[0]),
        seed_used=args.seed_start,
        model_path=None,
        training_hyperparameters={
            "training_seeds": parse_int_list(args.training_seeds),
            "maximum_timesteps": args.maximum_timesteps,
            "minimum_timesteps": args.minimum_timesteps,
            "eval_frequency_steps": args.eval_frequency_steps,
            "training_evaluation_episodes": args.training_evaluation_episodes,
            "td3_device": args.td3_device,
            "ppo_device": args.ppo_device,
            "evaluation_device": args.evaluation_device,
        },
        metrics_json_path=str(metrics_json_path),
        metrics_csv_path=str(metrics_csv_path),
        plot_paths=[str(plot_path), str(plot_path.with_suffix(".pdf"))],
        notes=payload["notes"],
    )
    report_writer.save_report(report, output_dir / "sensor_type_scalability_report.json")

    print("Saved", metrics_json_path, flush=True)
    print("Saved", metrics_csv_path, flush=True)
    print("Saved", per_trial_csv_path, flush=True)
    print("Saved", plot_path, flush=True)
    print("Saved", plot_path.with_suffix(".pdf"), flush=True)


def main() -> None:
    """CLI entry point."""
    args = parse_arguments()
    sensitivity_results = run_scalability_experiment(args)
    save_outputs(args, sensitivity_results)


if __name__ == "__main__":
    main()
