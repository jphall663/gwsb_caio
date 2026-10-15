# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import html
import mimetypes
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import CourseRecord, SourceRecord
from .util import stable_id


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        elif not self.hidden_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif not self.hidden_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n", value)
        return value.strip()


def html_to_text(content: bytes, charset: str = "utf-8") -> str:
    parser = _TextExtractor()
    parser.feed(content.decode(charset, errors="replace"))
    return parser.text()


def pdf_to_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        pages.append(f"\n--- Page {number} ---\n{page.extract_text() or ''}")
    return "".join(pages).strip()


def _extension(content_type: str, url: str) -> str:
    if "pdf" in content_type.lower() or urlparse(url).path.lower().endswith(".pdf"):
        return ".pdf"
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed if guessed in {".html", ".htm", ".txt"} else ".html"


def retrieve_url(
    url: str,
    artifact_dir: Path,
    *,
    retries: int = 2,
    timeout: int = 30,
) -> SourceRecord:
    record = SourceRecord(url=url)
    request = Request(url, headers={"User-Agent": "GWSB-Syllabus-Research/0.1 (+public course research)"})
    last_error = ""
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                final_url = response.geturl()
            identifier = stable_id(final_url)
            extension = _extension(content_type, final_url)
            raw_path = artifact_dir / "raw" / f"{identifier}{extension}"
            text_path = artifact_dir / "text" / f"{identifier}.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(content)
            text = pdf_to_text(content) if extension == ".pdf" else html_to_text(content, charset)
            text_path.write_text(text, encoding="utf-8")
            record.url = final_url
            record.retrieved_at = datetime.now(timezone.utc).isoformat()
            record.content_type = content_type
            record.artifact_path = str(raw_path)
            record.text_path = str(text_path)
            record.status = "retrieved" if text.strip() else "extraction_failed"
            if not text.strip():
                record.error = "No extractable text found."
            return record
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    record.status = "inaccessible"
    record.error = last_error
    record.retrieved_at = datetime.now(timezone.utc).isoformat()
    return record


def retrieve_courses(courses: list[CourseRecord], artifact_dir: Path, limits: dict) -> None:
    for course in courses:
        course.sources = [
            retrieve_url(
                url,
                artifact_dir,
                retries=int(limits.get("max_retrieval_retries", 2)),
                timeout=int(limits.get("request_timeout_seconds", 30)),
            )
            for url in course.source_urls
        ]
