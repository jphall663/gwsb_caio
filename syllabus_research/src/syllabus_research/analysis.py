# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .budget import BudgetTracker
from .config import ResearchConfig
from .llm import require_completed, response_call, response_dump, response_text
from .models import CourseRecord, CoverageEvidence
from .util import extract_json


def _evidence_packet(course: CourseRecord, max_characters: int) -> str:
    remaining = max_characters
    parts: list[str] = []
    for source in course.sources:
        if source.status != "retrieved" or not source.text_path:
            continue
        text = Path(source.text_path).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        numbered = "\n".join(f"L{number}: {line}" for number, line in enumerate(lines, 1))
        segment = f"SOURCE URL: {source.url}\n{numbered}"
        if len(segment) > remaining:
            segment = segment[:remaining] + "\n[TRUNCATED BY APPLICATION]"
        parts.append(segment)
        remaining -= len(segment)
        if remaining <= 0:
            break
    return "\n\n===== NEXT SOURCE =====\n\n".join(parts)


def build_analysis_prompt(course: CourseRecord, config: ResearchConfig) -> str:
    packet = _evidence_packet(
        course,
        int(config.limits.get("max_input_characters_per_source", 80000)),
    )
    schema = {
        "title": "course title exactly evidenced",
        "provider": "provider exactly evidenced",
        "instructor": "instructor or empty string",
        "audience": "audience or empty string",
        "prerequisites": "prerequisites or empty string",
        "language": "language",
        "date_evidence": "brief explicit excerpt or empty string",
        "date_iso": "YYYY-MM-DD or YYYY-MM or empty string",
        "date_source_location": "URL and line(s) or empty string",
        "ordered_outline": ["ordered item"],
        "rigor_evidence": "specific evidence",
        "audience_fit_evidence": "specific evidence",
        "coverage": [
            {
                "topic_id": "configured topic id",
                "status": "explicit|inferred|not_evidenced",
                "excerpt": "brief supporting excerpt or empty string",
                "source_url": "source URL or empty string",
                "source_location": "line(s)/section or empty string",
                "explanation": "why the status is warranted",
            }
        ],
        "additional_topics": [
            {
                "topic": "major topic not already configured",
                "evidence": "brief excerpt",
                "source_url": "URL",
                "source_location": "line(s)/section",
                "relevance": "relevance to master's business analytics students",
                "evidence_status": "explicit|inferred",
            }
        ],
        "uncertainties": ["material uncertainty"],
    }
    return f"""Analyze the retrieved public course materials below using only the supplied text.
Candidate metadata is a lead, not evidence: {json.dumps(course.to_dict(), ensure_ascii=False)}
Configured topics: {json.dumps(config.topics, ensure_ascii=False)}

Use `explicit` only for direct wording or an unmistakably synonymous outline item. Use
`inferred` only when the inference has concrete supporting text. Use `not_evidenced` when the
available material does not establish coverage; this means unknown, not absent. Include every
configured topic exactly once. Preserve outline order. Do not turn navigation, promotional
copy, testimonials, copyright years, or retrieval timestamps into course/date evidence.
Date evidence must describe the course offering or an explicit material publication/update.
Identify additional substantive technical/application topics that could fill curriculum gaps;
do not duplicate configured topics. Mark responsible-use material but remember it is taught in
another class.

Return JSON only matching this shape:
{json.dumps(schema, indent=2)}

RETRIEVED MATERIALS
===================
{packet}
"""


def analysis_schema(config: ResearchConfig) -> dict[str, Any]:
    topic_ids = [topic["id"] for topic in config.topics]
    coverage_item = {
        "type": "object",
        "properties": {
            "topic_id": {"type": "string", "enum": topic_ids},
            "status": {"type": "string", "enum": ["explicit", "inferred", "not_evidenced"]},
            "excerpt": {"type": "string"},
            "source_url": {"type": "string"},
            "source_location": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["topic_id", "status", "excerpt", "source_url", "source_location", "explanation"],
        "additionalProperties": False,
    }
    additional_topic = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "evidence": {"type": "string"},
            "source_url": {"type": "string"},
            "source_location": {"type": "string"},
            "relevance": {"type": "string"},
            "evidence_status": {"type": "string", "enum": ["explicit", "inferred"]},
        },
        "required": ["topic", "evidence", "source_url", "source_location", "relevance", "evidence_status"],
        "additionalProperties": False,
    }
    scalar_fields = {
        name: {"type": "string"}
        for name in (
            "title", "provider", "instructor", "audience", "prerequisites", "language",
            "date_evidence", "date_iso", "date_source_location", "rigor_evidence", "audience_fit_evidence",
        )
    }
    properties: dict[str, Any] = {
        **scalar_fields,
        "ordered_outline": {"type": "array", "items": {"type": "string"}},
        "coverage": {
            "type": "array",
            "items": coverage_item,
            "minItems": len(topic_ids),
            "maxItems": len(topic_ids),
        },
        "additional_topics": {"type": "array", "items": additional_topic},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def analyze_course(
    course: CourseRecord,
    config: ResearchConfig,
    tracker: BudgetTracker,
    record_response: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    if not any(source.status == "retrieved" for source in course.sources):
        course.uncertainties.append("No source could be retrieved and extracted.")
        return None
    response = response_call(
        tracker=tracker,
        prompt=build_analysis_prompt(course, config),
        model=config.model("analysis"),
        reasoning_effort=config.reasoning_effort,
        instructions="You are an evidence auditor comparing public course outlines.",
        max_output_tokens=int(config.limits.get("max_output_tokens", 12000)),
        response_schema=analysis_schema(config),
    )
    dump = response_dump(response)
    if record_response:
        record_response(dump)
    require_completed(response, f"analysis for {course.course_id}")
    result = extract_json(response_text(response))
    for field in ("title", "provider", "instructor", "audience", "prerequisites", "language"):
        value = result.get(field)
        if value:
            setattr(course, field, str(value))
    for field in ("date_evidence", "date_iso", "date_source_location"):
        if result.get(field):
            setattr(course, field, str(result[field]))
    course.outline = [str(value) for value in result.get("ordered_outline", []) if str(value).strip()]
    course.rigor_evidence = str(result.get("rigor_evidence", ""))
    course.audience_fit_evidence = str(result.get("audience_fit_evidence", ""))
    allowed_statuses = {"explicit", "inferred", "not_evidenced"}
    by_topic = {str(item.get("topic_id")): item for item in result.get("coverage", [])}
    course.coverage = []
    for topic in config.topics:
        item = by_topic.get(topic["id"], {})
        status = str(item.get("status", "not_evidenced"))
        if status not in allowed_statuses:
            status = "not_evidenced"
        course.coverage.append(
            CoverageEvidence(
                topic_id=topic["id"],
                status=status,
                excerpt=str(item.get("excerpt", "")),
                source_url=str(item.get("source_url", "")),
                source_location=str(item.get("source_location", "")),
                explanation=str(item.get("explanation", "")),
            )
        )
    course.additional_topics = [dict(item) for item in result.get("additional_topics", [])]
    course.uncertainties.extend(str(value) for value in result.get("uncertainties", []))
    return dump
