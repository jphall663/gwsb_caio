# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

from .config import ResearchConfig
from .models import CourseRecord


def subtract_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 - months
    year, zero_month = divmod(index, 12)
    month = zero_month + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def parse_course_date(value: str) -> date | None:
    value = value.strip()
    try:
        if len(value) == 10:
            return date.fromisoformat(value)
        if len(value) == 7:
            year, month = (int(part) for part in value.split("-"))
            return date(year, month, 1)
    except (ValueError, TypeError):
        return None
    return None


def _matches_override(course: CourseRecord, values: list[Any]) -> bool:
    haystacks = {course.course_id.lower(), course.canonical_url.lower(), course.title.lower()}
    for item in values:
        needle = str(item.get("url") or item.get("title") or item).lower() if isinstance(item, dict) else str(item).lower()
        if needle and any(needle == value or needle in value for value in haystacks):
            return True
    return False


def select_courses(courses: list[CourseRecord], config: ResearchConfig) -> dict[str, Any]:
    data = config.data
    selection = data["selection"]
    cutoff = subtract_months(config.as_of_date, int(data["recency_months"]))
    allowed = {value.lower() for value in selection["allowed_provider_types"]}
    minimum_topics = int(selection["minimum_substantive_topics"])

    eligible: list[CourseRecord] = []
    for course in courses:
        if course.duplicate_of:
            continue
        if _matches_override(course, data.get("explicit_exclusions", [])):
            course.selection_status = "excluded"
            course.exclusion_reason = "Explicit exclusion in the research profile."
            continue
        inclusion_override = _matches_override(course, data.get("explicit_inclusions", []))
        course.inclusion_override = inclusion_override
        retrieved = any(source.status == "retrieved" for source in course.sources)
        dated = parse_course_date(course.date_iso)
        current = dated is not None and cutoff <= dated <= config.as_of_date
        topic_count = sum(item.status in {"explicit", "inferred"} for item in course.coverage)
        provider_ok = course.provider_type.lower() in allowed
        language_ok = course.language.lower().startswith(data["language"].lower())
        substantive = len(course.outline) >= 3 and topic_count >= minimum_topics
        failures = []
        if not retrieved:
            failures.append("no accessible extracted outline")
        if not provider_ok:
            failures.append("provider type outside configured criteria")
        if not language_ok:
            failures.append("not evidenced as English-language")
        if not substantive:
            failures.append("insufficient substantive outline/topic evidence")
        if not course.rigor_evidence.strip():
            failures.append("insufficient rigor evidence")
        if not course.audience_fit_evidence.strip():
            failures.append("insufficient audience-fit evidence")
        if dated is None:
            failures.append("no qualifying explicit date evidence")
        elif not current:
            failures.append(f"date outside rolling window beginning {cutoff.isoformat()}")

        if inclusion_override:
            course.selection_status = "selected"
            course.override_outside_criteria = bool(failures)
            course.selection_rationale = "Explicit inclusion override. " + (
                "Outside criteria: " + "; ".join(failures) if failures else "Also meets normal criteria."
            )
            eligible.append(course)
        elif not failures:
            course.selection_status = "eligible"
            course.selection_rationale = (
                f"Public, current outline with {topic_count} evidenced configured topics and "
                "sufficient audience/rigor evidence."
            )
            eligible.append(course)
        elif dated is None and retrieved:
            course.selection_status = "unresolved"
            course.exclusion_reason = "; ".join(failures)
        else:
            course.selection_status = "excluded"
            course.exclusion_reason = "; ".join(failures)

    eligible.sort(
        key=lambda course: (
            course.inclusion_override,
            sum(item.status == "explicit" for item in course.coverage),
            sum(item.status == "inferred" for item in course.coverage),
            course.date_iso,
            len(course.outline),
        ),
        reverse=True,
    )
    maximum = int(data["selection_target_max"])
    selected_so_far = 0
    for course in eligible:
        if course.inclusion_override or selected_so_far < maximum:
            course.selection_status = "selected"
            selected_so_far += 1
        else:
            course.selection_status = "excluded"
            course.exclusion_reason = f"Eligible but below the top-{maximum} configured selection cap."
    selected_count = sum(course.selection_status == "selected" for course in courses)
    minimum = int(data["selection_target_min"])
    return {
        "cutoff_date": cutoff.isoformat(),
        "selected_count": selected_count,
        "target_min": minimum,
        "target_max": maximum,
        "shortfall": max(0, minimum - selected_count),
    }


def aggregate_gaps(courses: list[CourseRecord]) -> list[dict[str, Any]]:
    selected = [course for course in courses if course.selection_status == "selected"]
    groups: dict[str, dict[str, Any]] = {}
    for course in selected:
        for item in course.additional_topics:
            name = str(item.get("topic", "")).strip()
            if not name:
                continue
            key = name.casefold()
            group = groups.setdefault(
                key,
                {
                    "topic": name,
                    "supporting_courses": [],
                    "evidence": [],
                    "relevance": [],
                    "evidence_statuses": [],
                },
            )
            group["supporting_courses"].append(f"{course.provider}: {course.title}")
            evidence = str(item.get("evidence", ""))
            location = str(item.get("source_location", ""))
            if evidence:
                group["evidence"].append(f"{evidence} ({location})".strip())
            if item.get("relevance"):
                group["relevance"].append(str(item["relevance"]))
            group["evidence_statuses"].append(str(item.get("evidence_status", "inferred")))
    output = []
    denominator = len(selected)
    for group in groups.values():
        unique_courses = list(dict.fromkeys(group["supporting_courses"]))
        output.append(
            {
                "topic": group["topic"],
                "supporting_courses": unique_courses,
                "course_count": len(unique_courses),
                "prevalence": len(unique_courses) / denominator if denominator else 0,
                "evidence": group["evidence"],
                "recommendation": list(dict.fromkeys(group["relevance"])),
                "evidence_status": "explicit" if "explicit" in group["evidence_statuses"] else "inferred",
                "note": "Frequency describes this collection; it does not establish curricular importance.",
            }
        )
    return sorted(output, key=lambda item: (-item["course_count"], item["topic"].casefold()))
