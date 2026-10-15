# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from syllabus_research.dedup import deduplicate
from syllabus_research.models import CourseRecord, SourceRecord


def test_repeated_offerings_do_not_inflate_course_count():
    older = CourseRecord(
        "old", "Applied LLMs Fall 2025", provider="Example University",
        canonical_url="https://example.edu/llms/2025", date_iso="2025-09",
        sources=[SourceRecord(url="https://example.edu/llms/2025", status="retrieved")],
    )
    newer = CourseRecord(
        "new", "Applied LLMs Fall 2026", provider="Example University",
        canonical_url="https://example.edu/llms/2026", date_iso="2026-09",
        sources=[SourceRecord(url="https://example.edu/llms/2026", status="retrieved")],
    )
    deduplicate([older, newer])
    assert older.duplicate_of == "new"
    assert older.selection_status == "excluded"
    assert newer.duplicate_of == ""
