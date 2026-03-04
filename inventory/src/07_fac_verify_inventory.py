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
# Verify dat/fac_ai_mentions_joined_summary.csv after pipeline run !!!        #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import csv
import re
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

### constants #################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
DEFAULT_INPUT_CSV = DATA_DIR / "fac_ai_mentions_joined_summary.csv"

configure_logging()
logger = get_logger(__name__)

### helpers ###################################################################

def parse_hits(value):
    try:
        return int(value)
    except Exception:
        return 0

def split_matches(value):
    value = (value or "").strip()
    if not value:
        return []
    return [token.strip() for token in value.split(";") if token.strip()]

def normalize_ws(value):
    return re.sub(r"\s+", " ", (value or "").strip())

def normalize_name(name):
    return normalize_ws(name).lower()

def name_last_token(name):
    tokens = re.findall(r"[A-Za-z0-9]+", normalize_ws(name))
    if not tokens:
        return ""
    return tokens[-1].lower()

def contains_token(haystack, needle):
    return needle.lower() in (haystack or "").lower()

def contains_name_phrase(haystack, full_name):
    hay = (haystack or "").lower()
    phrase = normalize_name(full_name)
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, hay) is not None

def nonblank(value):
    return bool((value or "").strip())

### checks ####################################################################

def check_total_hits(row, row_idx, name):
    total = parse_hits(row.get("total_hits", ""))
    calc = (
        parse_hits(row.get("fac_profile_num_hits", ""))
        + parse_hits(row.get("per_site_num_hits", ""))
        + parse_hits(row.get("cv_num_hits", ""))
        + parse_hits(row.get("manual_num_hits", ""))
    )
    if total != calc:
        logger.warning(
            "Row %d (%s): total_hits mismatch (total_hits=%s, expected=%s)",
            row_idx,
            name,
            total,
            calc,
        )
        return 1
    return 0

def check_basic_structure(row, row_idx, name):
    issues = 0

    if parse_hits(row.get("fac_profile_num_hits", "")) > 0:
        for key in ("fac_profile_url", "fac_profile_matches", "fac_profile_snippets"):
            if not nonblank(row.get(key, "")):
                logger.warning("Row %d (%s): %s is blank with fac_profile_num_hits > 0", row_idx, name, key)
                issues += 1

    if parse_hits(row.get("per_site_num_hits", "")) > 0:
        for key in ("per_site_url", "per_site_matches", "per_site_snippets"):
            if not nonblank(row.get(key, "")):
                logger.warning("Row %d (%s): %s is blank with per_site_num_hits > 0", row_idx, name, key)
                issues += 1

    if parse_hits(row.get("cv_num_hits", "")) > 0:
        for key in ("cv_filename", "cv_matches", "cv_snippets"):
            if not nonblank(row.get(key, "")):
                logger.warning("Row %d (%s): %s is blank with cv_num_hits > 0", row_idx, name, key)
                issues += 1

    if parse_hits(row.get("manual_num_hits", "")) > 0:
        for key in ("manual_filename", "manual_matches", "manual_snippets"):
            if not nonblank(row.get(key, "")):
                logger.warning("Row %d (%s): %s is blank with manual_num_hits > 0", row_idx, name, key)
                issues += 1

    return issues

def check_matches_in_snippets(row, row_idx, name):
    issues = 0

    checks = [
        ("fac_profile_num_hits", "fac_profile_matches", "fac_profile_snippets"),
        ("per_site_num_hits", "per_site_matches", "per_site_snippets"),
        ("cv_num_hits", "cv_matches", "cv_snippets"),
        ("manual_num_hits", "manual_matches", "manual_snippets"),
    ]

    for hits_key, matches_key, snippets_key in checks:
        if parse_hits(row.get(hits_key, "")) <= 0:
            continue
        snippets = row.get(snippets_key, "")
        for token in split_matches(row.get(matches_key, "")):
            if not contains_token(snippets, token):
                logger.warning(
                    "Row %d (%s): token '%s' from %s not found in %s",
                    row_idx,
                    name,
                    token,
                    matches_key,
                    snippets_key,
                )
                issues += 1

    return issues

def check_same_row_name_consistency(row, row_idx, name):
    issues = 0
    last_name = name_last_token(name)
    if not last_name:
        logger.warning("Row %d (%s): could not derive last name token", row_idx, name)
        return 1

    if parse_hits(row.get("fac_profile_num_hits", "")) > 0:
        if not contains_token(row.get("fac_profile_url", ""), last_name):
            logger.warning("Row %d (%s): last name '%s' not found in fac_profile_url", row_idx, name, last_name)
            issues += 1
        if not contains_token(row.get("fac_profile_snippets", ""), last_name):
            logger.warning("Row %d (%s): last name '%s' not found in fac_profile_snippets", row_idx, name, last_name)
            issues += 1

    if parse_hits(row.get("per_site_num_hits", "")) > 0:
        if not contains_token(row.get("per_site_snippets", ""), last_name):
            logger.warning("Row %d (%s): last name '%s' not found in per_site_snippets", row_idx, name, last_name)
            issues += 1

    if parse_hits(row.get("cv_num_hits", "")) > 0:
        if not contains_token(row.get("cv_snippets", ""), last_name):
            logger.warning("Row %d (%s): last name '%s' not found in cv_snippets", row_idx, name, last_name)
            issues += 1

    if parse_hits(row.get("manual_num_hits", "")) > 0:
        if not contains_token(row.get("manual_snippets", ""), last_name):
            logger.warning("Row %d (%s): last name '%s' not found in manual_snippets", row_idx, name, last_name)
            issues += 1

    if not contains_token(row.get("summary", ""), last_name):
        logger.warning("Row %d (%s): last name '%s' not found in summary", row_idx, name, last_name)
        issues += 1

    return issues

def check_cross_row_name_leakage(rows):
    issues = 0
    names = [normalize_ws(row.get("name", "")) for row in rows if normalize_ws(row.get("name", ""))]
    unique_names = sorted(set(names), key=str.lower)

    fields = [
        "fac_profile_url",
        "fac_profile_snippets",
        "per_site_snippets",
        "cv_snippets",
        "manual_snippets",
        "summary",
    ]

    for row_idx, row in enumerate(rows, 1):
        this_name = normalize_ws(row.get("name", ""))
        if not this_name:
            continue
        merged_text = " ".join((row.get(field, "") or "") for field in fields)
        for other_name in unique_names:
            if normalize_name(other_name) == normalize_name(this_name):
                continue
            if contains_name_phrase(merged_text, other_name):
                logger.warning(
                    "Row %d (%s): contains other row name '%s' in verification fields",
                    row_idx,
                    this_name,
                    other_name,
                )
                issues += 1
    return issues

### main entrypoint ###########################################################

def run_verify(input_csv=DEFAULT_INPUT_CSV):
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    with input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    warnings_count = 0
    for row_idx, row in enumerate(rows, 1):
        name = normalize_ws(row.get("name", ""))
        warnings_count += check_total_hits(row, row_idx, name)
        warnings_count += check_basic_structure(row, row_idx, name)
        warnings_count += check_matches_in_snippets(row, row_idx, name)
        warnings_count += check_same_row_name_consistency(row, row_idx, name)

    warnings_count += check_cross_row_name_leakage(rows)

    logger.info(
        "Verification completed for %d rows with %d warning(s).",
        len(rows),
        warnings_count,
    )

if __name__ == "__main__":
    run_verify()
