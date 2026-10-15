# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from syllabus_research.budget import BudgetTracker
from syllabus_research.config import ResearchConfig, load_config
from syllabus_research import discovery
from syllabus_research.models import RunUsage


PROFILE = Path(__file__).parents[1] / "config" / "profiles" / "llm_business_analytics.json"


class FakeResponse:
    def __init__(self, text: str, *, status: str = "completed"):
        self.output_text = text
        self.status = status
        self.incomplete_details = SimpleNamespace(reason="max_output_tokens")

    def model_dump(self, mode="json"):
        return {"status": self.status, "output_text": self.output_text}


def _config() -> ResearchConfig:
    base = load_config(PROFILE)
    data = deepcopy(base.data)
    data["as_of_date"] = "2026-10-15"
    return ResearchConfig(base.path, data)


def _candidate(url: str) -> dict:
    return {
        "title": "Applied LLMs",
        "provider": "Example University",
        "provider_type": "university",
        "instructor": "",
        "audience": "Graduate students",
        "canonical_url": url,
        "source_urls": [],
        "date_evidence": "Fall 2026",
        "date_iso": "2026-09",
        "date_source_location": "Course page",
        "discovery_evidence": "Detailed weekly outline",
    }


def test_discovery_uses_strict_schema_and_records_before_parsing(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return FakeResponse(json.dumps({"candidates": [_candidate("https://example.edu/llms")]}))

    monkeypatch.setattr(discovery, "response_call", fake_call)
    recorded = []
    courses, errors = discovery.discover(
        _config(), BudgetTracker(5, RunUsage()), max_candidates=2,
        record_response=lambda name, value: recorded.append((name, value)),
    )
    assert errors == []
    assert len(courses) == 1
    assert captured["response_schema"]["additionalProperties"] is False
    assert captured["response_schema"]["properties"]["candidates"]["maxItems"] == 2
    assert recorded[0][0] == "discovery_batch_01"


def test_incomplete_discovery_is_preserved_and_reported(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "response_call",
        lambda **kwargs: FakeResponse('{"candidates":[', status="incomplete"),
    )
    recorded = []
    with pytest.raises(RuntimeError, match="Raw responses were preserved"):
        discovery.discover(
            _config(), BudgetTracker(5, RunUsage()), max_candidates=2,
            record_response=lambda name, value: recorded.append((name, value)),
        )
    assert recorded == [
        ("discovery_batch_01", {"status": "incomplete", "output_text": '{"candidates":['})
    ]
