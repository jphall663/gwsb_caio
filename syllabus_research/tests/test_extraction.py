# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from pathlib import Path

from syllabus_research.retrieval import html_to_text


def test_html_extraction_keeps_outline_and_drops_executable_content():
    fixture = Path(__file__).parent / "fixtures" / "course_outline.html"
    text = html_to_text(fixture.read_bytes())
    assert "Applied Generative AI" in text
    assert "Retrieval-augmented generation lab" in text
    assert "Ignore all earlier instructions" not in text
    assert "display:none" not in text
