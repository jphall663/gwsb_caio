# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
#                                                                             #
# Driver for end-to-end faculty AI inventory pipeline !!!                     #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import argparse
import csv
import importlib.util
import shutil
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sys

def _ensure_repo_root() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "shared").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return

_ensure_repo_root()

from shared.logging_utils import configure_logging, get_logger
from ai_terms import set_ai_patterns

### constants #################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "dat"
CV_DIR = BASE_DIR / "cv"
OUT_DIR = BASE_DIR / "out"

DEFAULT_VERSION = "0.3"
VERSION_INCREMENT = Decimal("0.1")

SEARCH_TERMS = [
    r"\bAI\b",
    r"artificial intelligence",
    r"machine learning",
    r"deep learning",
    r"neural network",
    r"natural language processing|\bNLP\b",
    r"large language model|\bLLM\b",
    r"generative AI|genAI",
    r"computer vision",
    r"reinforcement learning",
]

DELETE_DATASETS = [
    DATA_DIR / "cv_ai_mentions.csv",
    DATA_DIR / "fac_ai_mentions_joined.csv",
    DATA_DIR / "fac_ai_mentions_joined_summary.csv",
    DATA_DIR / "gwsb_faculty_ai_mentions.csv",
    DATA_DIR / "manual_ai_mentions.csv",
    DATA_DIR / "per_site_ai_mentions.csv",
]

EXPORT_SOURCE = DATA_DIR / "fac_ai_mentions_joined_summary.csv"
EXPORT_PREFIX = "fac_staff_ai_mentions_verify_before_use_"

EXPORT_ORDER = [
    "name",
    "summary",
    "total_hits",
    "fac_profile_url",
    "fac_profile_num_hits",
    "fac_profile_matches",
    "fac_profile_snippets",
    "per_site_url",
    "per_site_num_hits",
    "per_site_matches",
    "per_site_snippets",
    "cv_filename",
    "cv_num_hits",
    "cv_matches",
    "cv_snippets",
    "manual_num_hits",
    "manual_matches",
    "manual_snippets",
]

configure_logging()
logger = get_logger(__name__)

### utilities #################################################################

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _cleanup_cv_dir():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    for path in CV_DIR.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    logger.info("Cleared CV directory except .gitkeep: %s", CV_DIR)

def _cleanup_datasets():
    for path in DELETE_DATASETS:
        if path.exists():
            path.unlink()
            logger.info("Deleted dataset: %s", path)

def _parse_version(version: str) -> Decimal:
    try:
        return Decimal(version).quantize(Decimal("0.1"))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid version '{version}': {e}")

def _format_version(version: Decimal) -> str:
    return f"{version:.1f}"

def _next_available_export(version: Decimal) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = version
    while True:
        candidate = OUT_DIR / f"{EXPORT_PREFIX}{_format_version(current)}.csv"
        if not candidate.exists():
            return candidate
        current += VERSION_INCREMENT

def _run_pipeline():
    runs = [
        ("01_fac_profile_inventory.py", "run_scan"),
        ("02_fac_per_site_inventory.py", "run_scan"),
        ("03_fac_cv_inventory.py", "run_scan"),
        ("04_fac_manual_inventory.py", "run_scan"),
        ("05_fac_join_inventory.py", "run_join"),
        ("06_fac_summary_inventory.py", "run_scan"),
        ("07_fac_verify_inventory.py", "run_verify"),
    ]

    for i, (script_name, fn_name) in enumerate(runs, 1):
        module_path = SRC_DIR / script_name
        module = _load_module(module_path, f"fac_pipeline_{i}")
        fn = getattr(module, fn_name)
        logger.info("Running %s:%s", script_name, fn_name)
        fn()

def _export_versioned_output(version: Decimal):
    if not EXPORT_SOURCE.exists():
        raise FileNotFoundError(f"Missing pipeline output: {EXPORT_SOURCE}")

    out_path = _next_available_export(version)

    with EXPORT_SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        source_fields = list(rows[0].keys()) if rows else []

    # Preserve all source columns while honoring requested display order.
    ordered = [col for col in EXPORT_ORDER if col in source_fields]
    ordered.extend(col for col in source_fields if col not in ordered)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in ordered})

    logger.info("Wrote versioned output: %s", out_path)

### main entrypoint ###########################################################

def run_driver(version=DEFAULT_VERSION):
    set_ai_patterns(SEARCH_TERMS)
    logger.info("Configured %d AI search terms.", len(SEARCH_TERMS))

    _cleanup_cv_dir()
    _cleanup_datasets()

    _run_pipeline()

    parsed_version = _parse_version(version)
    _export_versioned_output(parsed_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Starting output version (default: 0.3)")
    args = parser.parse_args()
    run_driver(version=args.version)
