"""Train TD3 on the defective RSU zone environment."""
from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from leader_dt.config import SimulationConfig, Td3TrainingConfig
from leader_dt.rl.td3_agent import Td3Trainer


def main() -> None:
    trainer = Td3Trainer(SimulationConfig(random_seed=1), Td3TrainingConfig())
    model = trainer.train()
    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model, str(output_dir / "td3_exact_pair_zone_b"))


if __name__ == "__main__":
    main()
