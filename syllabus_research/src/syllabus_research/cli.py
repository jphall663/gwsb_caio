# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PROFILE = PACKAGE_ROOT / "config" / "profiles" / "llm_business_analytics.json"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared.logging_utils import configure_logging

from .config import load_config, load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover, verify, compare, and export public course outlines to Excel."
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Research profile JSON.")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs", help="Generated run directory root.")
    parser.add_argument("--resume", type=Path, help="Existing run directory containing state.json.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Limit candidates for a smoke run. Normal research runs should omit this option.",
    )
    parser.add_argument("--env-file", type=Path, default=REPOSITORY_ROOT / ".env", help="Environment file.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_candidates is not None and args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive")
    load_dotenv(args.env_file)
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    from .pipeline import run_pipeline

    config = load_config(args.profile)
    workbook = run_pipeline(
        config,
        output_root=args.output,
        resume_dir=args.resume,
        max_candidates=args.max_candidates,
    )
    print(workbook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
