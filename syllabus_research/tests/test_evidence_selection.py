# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from copy import deepcopy
from pathlib import Path

from syllabus_research.config import ResearchConfig, load_config
from syllabus_research.models import CourseRecord, CoverageEvidence, SourceRecord
from syllabus_research.selection import select_courses


PROFILE = Path(__file__).parents[1] / "config" / "profiles" / "llm_business_analytics.json"


def _course(course_id: str, date_iso: str) -> CourseRecord:
    return CourseRecord(
        course_id,
        f"Course {course_id}",
        provider="Example University",
        provider_type="university",
        canonical_url=f"https://example.edu/{course_id}",
        source_urls=[f"https://example.edu/{course_id}"],
        date_iso=date_iso,
        date_evidence="Fall 2026",
        language="English",
        outline=["Transformers", "Prompting", "RAG", "Agents"],
        rigor_evidence="Applied lab and final project.",
        audience_fit_evidence="Designed for graduate business analytics students.",
        sources=[SourceRecord(url="https://example.edu", status="retrieved")],
        coverage=[
            CoverageEvidence("transformers", "explicit"),
            CoverageEvidence("prompting_icl", "explicit"),
            CoverageEvidence("rag", "inferred"),
            CoverageEvidence("evaluation", "not_evidenced"),
        ],
    )


def test_undated_course_stays_unresolved_and_not_evidenced_is_not_absent():
    base = load_config(PROFILE)
    data = deepcopy(base.data)
    data["as_of_date"] = "2026-10-15"
    data["selection_target_min"] = 1
    config = ResearchConfig(base.path, data)
    current = _course("current", "2026-09")
    undated = _course("undated", "")
    info = select_courses([current, undated], config)
    assert current.selection_status == "selected"
    assert undated.selection_status == "unresolved"
    assert "no qualifying explicit date evidence" in undated.exclusion_reason
    assert current.coverage[-1].status == "not_evidenced"
    assert info["shortfall"] == 0


def test_inclusion_override_is_selected_but_flagged_outside_criteria():
    base = load_config(PROFILE)
    data = deepcopy(base.data)
    data["as_of_date"] = "2026-10-15"
    data["selection_target_min"] = 1
    data["explicit_inclusions"] = [{"url": "https://example.edu/override"}]
    config = ResearchConfig(base.path, data)
    course = _course("override", "")
    select_courses([course], config)
    assert course.selection_status == "selected"
    assert course.inclusion_override is True
    assert course.override_outside_criteria is True


def test_inclusion_override_is_not_dropped_by_selection_cap():
    base = load_config(PROFILE)
    data = deepcopy(base.data)
    data["as_of_date"] = "2026-10-15"
    data["selection_target_min"] = 1
    data["selection_target_max"] = 1
    data["explicit_inclusions"] = [{"url": "https://example.edu/override"}]
    config = ResearchConfig(base.path, data)
    ordinary = _course("ordinary", "2026-09")
    override = _course("override", "")
    select_courses([ordinary, override], config)
    assert override.selection_status == "selected"
    assert ordinary.selection_status == "excluded"
