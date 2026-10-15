# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit, urlunsplit

from .models import CourseRecord


def normalize_text(value: str) -> str:
    value = re.sub(r"\b(?:spring|summer|fall|autumn|winter)\s+20\d{2}\b", "", value.lower())
    value = re.sub(r"\b20\d{2}\b", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def normalize_url(value: str) -> str:
    parsed = urlsplit(value)
    path = re.sub(r"/+$", "", parsed.path)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _quality(course: CourseRecord) -> tuple[int, str, int]:
    retrieved = sum(source.status == "retrieved" for source in course.sources)
    return (retrieved, course.date_iso, len(course.outline))


def deduplicate(courses: list[CourseRecord]) -> None:
    groups: list[list[CourseRecord]] = []
    for course in courses:
        title = normalize_text(course.title)
        provider = normalize_text(course.provider)
        url = normalize_url(course.canonical_url)
        matched = None
        for group in groups:
            representative = group[0]
            same_url = url and url == normalize_url(representative.canonical_url)
            same_course = (
                provider == normalize_text(representative.provider)
                and SequenceMatcher(None, title, normalize_text(representative.title)).ratio() >= 0.9
            )
            if same_url or same_course:
                matched = group
                break
        if matched is None:
            groups.append([course])
        else:
            matched.append(course)

    for group in groups:
        if len(group) < 2:
            continue
        winner = max(group, key=_quality)
        for course in group:
            if course is not winner:
                course.duplicate_of = winner.course_id
                course.selection_status = "excluded"
                course.exclusion_reason = f"Duplicate/version of {winner.title} ({winner.course_id})."
