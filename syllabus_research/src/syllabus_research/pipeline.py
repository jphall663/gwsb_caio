# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.logging_utils import get_logger

from .analysis import analyze_course
from .budget import BudgetTracker, CallLimitExceeded, SpendingLimitExceeded
from .config import ResearchConfig
from .dedup import deduplicate
from .discovery import discover
from .models import CourseRecord, RunUsage
from .retrieval import retrieve_url
from .selection import aggregate_gaps, select_courses
from .util import atomic_json_write
from .workbook import export_workbook


logger = get_logger(__name__)


def new_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / timestamp
    suffix = 1
    while path.exists():
        path = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    return path


def _state_payload(
    courses: list[CourseRecord],
    usage: RunUsage,
    completed_stages: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_stages": completed_stages,
        "limitations": limitations,
        "usage": usage.to_dict(),
        "courses": [course.to_dict() for course in courses],
    }


def run_pipeline(
    config: ResearchConfig,
    *,
    output_root: Path,
    resume_dir: Path | None = None,
    max_candidates: int | None = None,
) -> Path:
    run_dir = resume_dir.resolve() if resume_dir else new_run_dir(output_root.resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(exist_ok=True)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        courses = [CourseRecord.from_dict(item) for item in state.get("courses", [])]
        usage = RunUsage(**state.get("usage", {}))
        completed = list(state.get("completed_stages", []))
        limitations = list(state.get("limitations", []))
        logger.info("Resuming run %s after stages: %s", run_dir.name, ", ".join(completed) or "none")
    else:
        courses = []
        usage = RunUsage()
        completed = []
        limitations = []
        shutil.copy2(config.path, provenance_dir / "profile.json")
        atomic_json_write(
            provenance_dir / "run.json",
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "application_version": "0.1.0",
                "discovery_model": config.model("discovery"),
                "analysis_model": config.model("analysis"),
                "reasoning_effort": config.reasoning_effort,
                "max_candidates_override": max_candidates,
            },
        )

    tracker = BudgetTracker(
        int(config.limits.get("max_api_calls", 60)),
        usage,
        estimated_usd_limit=config.limits.get("estimated_usd_limit"),
        per_call_cost_reserve_usd=config.limits.get("per_call_cost_reserve_usd"),
        usd_per_million_input_tokens=config.limits.get("usd_per_million_input_tokens"),
        usd_per_million_output_tokens=config.limits.get("usd_per_million_output_tokens"),
        usd_per_web_search_call=config.limits.get("usd_per_web_search_call"),
    )

    def checkpoint() -> None:
        atomic_json_write(state_path, _state_payload(courses, usage, completed, limitations))

    if "discovery" not in completed:
        logger.info("Discovering public course outlines with built-in web search")
        discovery_dir = provenance_dir / "discovery_responses"
        discovery_dir.mkdir(exist_ok=True)
        courses, discovery_errors = discover(
            config,
            tracker,
            max_candidates=max_candidates,
            record_response=lambda name, value: atomic_json_write(discovery_dir / f"{name}.json", value),
        )
        limitations.extend(discovery_errors)
        completed.append("discovery")
        checkpoint()

    if "retrieval" not in completed:
        logger.info("Retrieving and extracting %d candidate(s)", len(courses))
        for course in courses:
            if course.sources:
                continue
            for url in course.source_urls:
                source = retrieve_url(
                    url,
                    run_dir / "artifacts",
                    retries=int(config.limits.get("max_retrieval_retries", 2)),
                    timeout=int(config.limits.get("request_timeout_seconds", 30)),
                )
                course.sources.append(source)
                checkpoint()
        completed.append("retrieval")
        checkpoint()

    if "analysis" not in completed:
        logger.info("Analyzing retrieved evidence")
        analysis_dir = provenance_dir / "analysis_responses"
        analysis_dir.mkdir(exist_ok=True)
        for course in courses:
            if course.coverage or not any(source.status == "retrieved" for source in course.sources):
                continue
            try:
                analyze_course(
                    course,
                    config,
                    tracker,
                    record_response=lambda value, course_id=course.course_id: atomic_json_write(
                        analysis_dir / f"{course_id}.json", value
                    ),
                )
            except (CallLimitExceeded, SpendingLimitExceeded) as exc:
                message = str(exc)
                if message not in limitations:
                    limitations.append(message)
                checkpoint()
                break
            except Exception as exc:  # per-source fault tolerance is intentional
                course.uncertainties.append(f"Analysis failed: {type(exc).__name__}: {exc}")
                logger.warning("Analysis failed for %s: %s", course.title, exc)
            checkpoint()
        completed.append("analysis")
        checkpoint()

    deduplicate(courses)
    selection_info = select_courses(courses, config)
    if selection_info["shortfall"]:
        limitations.append(
            f"Only {selection_info['selected_count']} strong matches qualified; "
            f"the configured minimum is {selection_info['target_min']}. Criteria were not relaxed."
        )
    limitations = list(dict.fromkeys(limitations))
    gaps = aggregate_gaps(courses)
    workbook_path = run_dir / "syllabus_comparison.xlsx"
    export_workbook(
        workbook_path,
        courses=courses,
        gaps=gaps,
        config=config,
        usage=usage,
        run_info={**selection_info, "limitations": limitations},
    )
    if "export" not in completed:
        completed.append("export")
    checkpoint()
    logger.info("Workbook written to %s", workbook_path)
    return workbook_path
