"""Save and load result objects."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any


class ResultsWriter:
    def save_json(self, data: dict, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self._make_json_safe(data), indent=2))

    def load_json(self, input_path: str | Path) -> dict:
        return json.loads(Path(input_path).read_text())

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
