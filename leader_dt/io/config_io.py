"""Configuration serialization helpers."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from typing import Any

from leader_dt.config import (
    CommunicationConfig,
    DataGenerationConfig,
    MonteCarloConfig,
    RoadConfig,
    SensitivityConfig,
    SimulationConfig,
    SystemConfig,
    Td3TrainingConfig,
)


def dataclass_to_dictionary(config_object: Any) -> dict:
    if not is_dataclass(config_object):
        raise TypeError("config_object must be a dataclass instance.")
    return asdict(config_object)


def save_config(config_object: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataclass_to_dictionary(config_object), indent=2))
    return output_path


def load_json_dictionary(input_path: str | Path) -> dict:
    return json.loads(Path(input_path).read_text())


def simulation_config_from_dictionary(config_dictionary: dict) -> SimulationConfig:
    return SimulationConfig(
        system=SystemConfig(**config_dictionary.get("system", {})),
        communication=CommunicationConfig(**config_dictionary.get("communication", {})),
        road=RoadConfig(**config_dictionary.get("road", {})),
        data_generation=DataGenerationConfig(**config_dictionary.get("data_generation", {})),
        random_seed=config_dictionary.get("random_seed"),
    )


def td3_training_config_from_dictionary(config_dictionary: dict) -> Td3TrainingConfig:
    return Td3TrainingConfig(**config_dictionary)


def monte_carlo_config_from_dictionary(config_dictionary: dict) -> MonteCarloConfig:
    return MonteCarloConfig(**config_dictionary)


def sensitivity_config_from_dictionary(config_dictionary: dict) -> SensitivityConfig:
    return SensitivityConfig(**config_dictionary)
