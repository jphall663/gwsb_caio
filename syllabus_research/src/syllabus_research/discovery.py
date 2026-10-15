# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from .budget import BudgetTracker, CallLimitExceeded, SpendingLimitExceeded
from .config import ResearchConfig
from .llm import require_completed, response_call, response_dump, response_text
from .models import CourseRecord
from .util import extract_json, stable_id


DISCOVERY_FOCUSES = (
    "University courses and required seed sources",
    "Reputable professional training and rigorous independent instructors",
    "Applied courses emphasizing RAG, agents, evaluation, and production systems",
    "Courses emphasizing prompting, context engineering, structured outputs, tools, and APIs",
)
logger = logging.getLogger(__name__)


def discovery_schema(max_candidates: int) -> dict[str, Any]:
    candidate = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "provider": {"type": "string"},
            "provider_type": {
                "type": "string",
                "enum": ["university", "professional training", "independent instructor"],
            },
            "instructor": {"type": "string"},
            "audience": {"type": "string"},
            "canonical_url": {"type": "string"},
            "source_urls": {"type": "array", "items": {"type": "string"}},
            "date_evidence": {"type": "string"},
            "date_iso": {"type": "string"},
            "date_source_location": {"type": "string"},
            "discovery_evidence": {"type": "string"},
        },
        "required": [
            "title", "provider", "provider_type", "instructor", "audience", "canonical_url",
            "source_urls", "date_evidence", "date_iso", "date_source_location", "discovery_evidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": candidate,
                "maxItems": max_candidates,
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def build_discovery_prompt(
    config: ResearchConfig,
    *,
    batch_size: int,
    focus: str,
    prior_urls: list[str],
) -> str:
    data = config.data
    return f"""Discover up to {batch_size} candidate public course syllabi, detailed outlines,
lesson indexes, or teaching repositories for this research profile.

This batch's focus: {focus}
Profile: {data['profile_name']}
Audience: {data['audience']}
Required topics: {json.dumps(data['topics'], ensure_ascii=False)}
Seed sources (search each required seed in the seed-focused batch): {json.dumps(data['seed_sources'], ensure_ascii=False)}
Explicit inclusions: {json.dumps(data.get('explicit_inclusions', []), ensure_ascii=False)}
Explicit exclusions: {json.dumps(data.get('explicit_exclusions', []), ensure_ascii=False)}
Provider types: {json.dumps(data['selection']['allowed_provider_types'])}
Language: {data['language']}
As-of date: {config.as_of_date.isoformat()}; rolling window: {data['recency_months']} months.
URLs already returned by earlier batches (do not repeat): {json.dumps(prior_urls, ensure_ascii=False)}

Search broadly within the batch focus, but return only plausible candidates with a directly
accessible outline or repository URL. Paid courses qualify only through public outlines. Relevant
modules in broader courses may qualify. Do not use copyright years, retrieval dates, popularity,
or search snippets as date or quality evidence. Include undated candidates with blank date fields
so the application can keep them unresolved. Do not claim a topic is absent from a partial outline.
Return the structured JSON result only. Use empty strings for unavailable scalar values.
"""


def _to_courses(values: list[dict[str, Any]]) -> list[CourseRecord]:
    courses: list[CourseRecord] = []
    for item in values:
        url = str(item.get("canonical_url", "")).strip()
        title = str(item.get("title", "")).strip()
        provider = str(item.get("provider", "")).strip()
        if not title or not url:
            continue
        urls = [url, *item.get("source_urls", [])]
        urls = list(dict.fromkeys(str(value).strip() for value in urls if str(value).strip()))
        courses.append(
            CourseRecord(
                course_id=stable_id(provider, title, url),
                title=title,
                provider=provider,
                provider_type=str(item.get("provider_type", "")),
                instructor=str(item.get("instructor", "")),
                audience=str(item.get("audience", "")),
                canonical_url=url,
                source_urls=urls,
                discovery_evidence=str(item.get("discovery_evidence", "")),
                date_evidence=str(item.get("date_evidence", "")),
                date_iso=str(item.get("date_iso", "")),
                date_source_location=str(item.get("date_source_location", "")),
            )
        )
    return courses


def discover(
    config: ResearchConfig,
    tracker: BudgetTracker,
    *,
    max_candidates: int | None = None,
    record_response: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[CourseRecord], list[str]]:
    limits = config.limits
    if max_candidates is not None:
        pool_target = max_candidates
        batch_size = max_candidates
        max_batches = 1
    else:
        pool_target = int(limits.get("candidate_pool_target", 40))
        batch_size = int(limits.get("discovery_batch_size", 10))
        max_batches = int(limits.get("discovery_max_batches", 4))
    if min(pool_target, batch_size, max_batches) < 1:
        raise ValueError("Discovery pool, batch size, and batch count must be positive.")

    courses: list[CourseRecord] = []
    seen_urls: set[str] = set()
    errors: list[str] = []
    for index in range(max_batches):
        if len(courses) >= pool_target:
            break
        requested = min(batch_size, pool_target - len(courses))
        focus = DISCOVERY_FOCUSES[index % len(DISCOVERY_FOCUSES)]
        response_name = f"discovery_batch_{index + 1:02d}"
        logger.info("Running %s (%s; up to %d candidates)", response_name, focus, requested)
        try:
            response = response_call(
                tracker=tracker,
                prompt=build_discovery_prompt(
                    config,
                    batch_size=requested,
                    focus=focus,
                    prior_urls=sorted(seen_urls),
                ),
                model=config.model("discovery"),
                reasoning_effort=config.reasoning_effort,
                instructions="You are a careful research librarian finding course-level primary sources.",
                tools=[{"type": "web_search"}],
                tool_choice="required",
                max_output_tokens=int(limits.get("max_output_tokens", 12000)),
                response_schema=discovery_schema(requested),
            )
        except (CallLimitExceeded, SpendingLimitExceeded):
            raise
        except Exception as exc:
            errors.append(f"{response_name} API call failed: {type(exc).__name__}: {exc}")
            logger.warning("%s", errors[-1])
            continue
        dump = response_dump(response)
        if record_response:
            record_response(response_name, dump)
        try:
            require_completed(response, response_name)
            parsed = extract_json(response_text(response))
            values = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            batch_courses = _to_courses(values)
            for course in batch_courses:
                if course.canonical_url in seen_urls:
                    continue
                seen_urls.add(course.canonical_url)
                courses.append(course)
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{response_name} failed: {type(exc).__name__}: {exc}")
            logger.warning("%s", errors[-1])

    if not courses:
        detail = "; ".join(errors) or "No candidates were returned."
        raise RuntimeError(f"All discovery batches failed. Raw responses were preserved. {detail}")
    return courses[:pool_target], errors
