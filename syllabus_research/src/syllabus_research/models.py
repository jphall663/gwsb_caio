# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceRecord:
    url: str
    title: str = ""
    retrieved_at: str = ""
    content_type: str = ""
    artifact_path: str = ""
    text_path: str = ""
    status: str = "pending"
    error: str = ""
    date_evidence: str = ""
    date_source_location: str = ""


@dataclass
class CoverageEvidence:
    topic_id: str
    status: str = "not_evidenced"
    excerpt: str = ""
    source_url: str = ""
    source_location: str = ""
    explanation: str = ""


@dataclass
class CourseRecord:
    course_id: str
    title: str
    provider: str = ""
    provider_type: str = ""
    instructor: str = ""
    audience: str = ""
    prerequisites: str = ""
    language: str = "English"
    canonical_url: str = ""
    source_urls: list[str] = field(default_factory=list)
    discovery_evidence: str = ""
    date_evidence: str = ""
    date_iso: str = ""
    date_source_location: str = ""
    outline: list[str] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    coverage: list[CoverageEvidence] = field(default_factory=list)
    additional_topics: list[dict[str, str]] = field(default_factory=list)
    selection_rationale: str = ""
    selection_status: str = "unresolved"
    exclusion_reason: str = ""
    uncertainties: list[str] = field(default_factory=list)
    inclusion_override: bool = False
    override_outside_criteria: bool = False
    duplicate_of: str = ""
    rigor_evidence: str = ""
    audience_fit_evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CourseRecord":
        values = dict(data)
        values["sources"] = [SourceRecord(**item) for item in values.get("sources", [])]
        values["coverage"] = [CoverageEvidence(**item) for item in values.get("coverage", [])]
        return cls(**values)


@dataclass
class RunUsage:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
