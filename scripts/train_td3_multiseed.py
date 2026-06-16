"""Train TD3 across multiple seeds."""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace
from leader_dt.config import SimulationConfig, Td3TrainingConfig
from leader_dt.evaluation.reporting import ReportWriter
from leader_dt.rl.td3_agent import Td3Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/models/multiseed")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_writer = ReportWriter()

    for seed in seeds:
        simulation_config = SimulationConfig(random_seed=seed)
        training_config = Td3TrainingConfig()
        if args.timesteps is not None:
            training_config = replace(training_config, total_timesteps=args.timesteps)
        trainer = Td3Trainer(simulation_config, training_config)
        model = trainer.train()
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_path = seed_dir / "td3_exact_pair_zone_b"
        trainer.save_model(model, str(model_path))
        report = report_writer.build_report(
            config_used=simulation_config,
            seed_used=seed,
            model_path=str(model_path),
            training_hyperparameters=training_config,
        )
        report_writer.save_report(report, seed_dir / "training_report.json")
        print("Saved seed", seed, "to", seed_dir)


if __name__ == "__main__":
    main()
