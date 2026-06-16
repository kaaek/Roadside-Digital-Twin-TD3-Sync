"""Experiment reporting helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import csv
import json
from typing import Any

from leader_dt.evaluation.monte_carlo import MonteCarloResult

@dataclass
class ExperimentReport:
    config_used: dict = field(default_factory=dict)
    seed_used: int | None = None
    model_path: str | None = None
    training_hyperparameters: dict = field(default_factory=dict)
    metrics_json_path: str | None = None
    metrics_csv_path: str | None = None
    plot_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dictionary(self) -> dict:
        return asdict(self)


class ReportWriter:
    def save_report(self, report: ExperimentReport, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dictionary(), indent=2))
        return output_path

    def save_metrics_json(self, metrics_dictionary: dict, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self._make_json_safe(metrics_dictionary), indent=2))
        return output_path

    def save_metrics_csv(self, rows: list[dict], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(rows) == 0:
            output_path.write_text("")
            return output_path
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with output_path.open("w", newline="") as file_object:
            writer = csv.DictWriter(file_object, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    def monte_carlo_results_to_rows(self, result_dictionary: dict[str, MonteCarloResult]) -> list[dict]:
        rows: list[dict] = []
        for policy_name, result in result_dictionary.items():
            mean_row = {"policy_name": policy_name, "statistic": "mean"}
            mean_row.update(result.metric_mean_dictionary)
            rows.append(mean_row)
            std_row = {"policy_name": policy_name, "statistic": "std"}
            std_row.update(result.metric_std_dictionary)
            rows.append(std_row)
        return rows

    def build_report(
        self,
        config_used: Any,
        seed_used: int | None = None,
        model_path: str | None = None,
        training_hyperparameters: Any | None = None,
        metrics_json_path: str | None = None,
        metrics_csv_path: str | None = None,
        plot_paths: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> ExperimentReport:
        return ExperimentReport(
            config_used=self._dataclass_or_dictionary(config_used),
            seed_used=seed_used,
            model_path=model_path,
            training_hyperparameters=self._dataclass_or_dictionary(training_hyperparameters) if training_hyperparameters is not None else {},
            metrics_json_path=metrics_json_path,
            metrics_csv_path=metrics_csv_path,
            plot_paths=plot_paths or [],
            notes=notes or [],
        )

    def _dataclass_or_dictionary(self, value: Any) -> dict:
        if value is None:
            return {}
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return value
        raise TypeError("Expected a dataclass or dictionary.")

    def _make_json_safe(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(item) for item in value]
        if hasattr(value, "__dict__"):
            return self._make_json_safe(value.__dict__)
        return value
