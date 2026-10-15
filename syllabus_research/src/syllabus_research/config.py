# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE file without overwriting exported variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ResearchConfig:
    path: Path
    data: dict[str, Any]

    @property
    def as_of_date(self) -> date:
        value = self.data.get("as_of_date")
        return date.fromisoformat(value) if value else date.today()

    @property
    def topics(self) -> list[dict[str, Any]]:
        return list(self.data["topics"])

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self.data.get("limits", {}))

    def model(self, stage: str) -> str:
        env = os.environ.get("SYLLABUS_RESEARCH_MODEL")
        return env or self.data["models"][stage]

    @property
    def reasoning_effort(self) -> str | None:
        return os.environ.get("SYLLABUS_RESEARCH_REASONING") or self.data.get("reasoning_effort")


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"profile_name", "topics", "seed_sources", "selection", "models", "limits"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Profile is missing required keys: {', '.join(missing)}")
    topic_ids = [topic["id"] for topic in data["topics"]]
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError("Topic ids must be unique.")
    minimum = int(data.get("selection_target_min", 20))
    maximum = int(data.get("selection_target_max", 30))
    if not 0 < minimum <= maximum:
        raise ValueError("selection_target_min/max are invalid.")
    return ResearchConfig(config_path, data)
