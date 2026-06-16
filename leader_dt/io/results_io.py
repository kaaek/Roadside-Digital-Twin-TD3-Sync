"""Save and load result objects."""

from __future__ import annotations

import json
from pathlib import Path


class ResultsWriter:
    def save_json(self, data: dict, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))

    def load_json(self, input_path: str | Path) -> dict:
        return json.loads(Path(input_path).read_text())