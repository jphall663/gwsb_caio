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
# Add OpenAI Key to config.py before running !!!                              #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import csv
import time
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
from shared.llms.openai_client import gpt_complete

### constants #################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"

DEFAULT_INPUT_CSV = DATA_DIR / "fac_ai_mentions_joined.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "fac_ai_mentions_joined_summary.csv"

configure_logging()
logger = get_logger(__name__)

### prompt utilities ##########################################################

def parse_hits(value):
    try:
        return int(value)
    except Exception:
        return 0

def build_source_block(label, matches, snippets):
    matches = (matches or "").strip()
    snippets = (snippets or "").strip()
    return f"Source: {label}\nMatches: {matches}\nSnippets: {snippets}".strip()

def build_prompt(name, blocks, sources):
    sources_text = ", ".join(sources)
    blocks_text = "\n\n".join(blocks)
    return (
        "You are summarizing AI-related work based only on the provided matches and snippets.\n"
        "Make the source of each summary explicit and do not add facts beyond the text.\n"
        "Use the exact template below.\n\n"
        f"Template:\n"
        f"{name}'s AI work was extracted from {sources_text}.\n\n"
        "Summary of <source>: <summary of match and snippets associated with the source>\n\n"
        "Sources:\n"
        f"{blocks_text}\n\n"
        "Output:\n"
    )

### main entrypoint ###########################################################

def run_scan(input_csv=DEFAULT_INPUT_CSV, output_csv=DEFAULT_OUTPUT_CSV, delay_s=0.5):
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    rows = []
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            sources = []
            blocks = []

            fac_hits = parse_hits(row.get("fac_profile_num_hits", ""))
            if fac_hits > 0:
                sources.append("fac_profile")
                blocks.append(build_source_block("fac_profile", row.get("fac_profile_matches"), row.get("fac_profile_snippets")))

            per_hits = parse_hits(row.get("per_site_num_hits", ""))
            if per_hits > 0:
                sources.append("per_site")
                blocks.append(build_source_block("per_site", row.get("per_site_matches"), row.get("per_site_snippets")))

            cv_hits = parse_hits(row.get("cv_num_hits", ""))
            if cv_hits > 0:
                sources.append("cv")
                blocks.append(build_source_block("cv", row.get("cv_matches"), row.get("cv_snippets")))

            manual_hits = parse_hits(row.get("manual_num_hits", ""))
            if manual_hits > 0:
                sources.append("manual")
                blocks.append(build_source_block("manual", row.get("manual_matches"), row.get("manual_snippets")))

            if not sources:
                row["summary"] = ""
                rows.append(row)
                continue

            prompt = build_prompt(name, blocks, sources)

            try:
                response = gpt_complete(prompt)
                content = response.choices[0].message.content.strip()
                row["summary"] = content
                logger.info("Summarized: %s", name)
            except Exception as e:
                logger.error("Summary failed for %s: %s", name, e)
                row["summary"] = ""

            rows.append(row)
            time.sleep(delay_s)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    if "summary" not in fieldnames:
        fieldnames.append("summary")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    run_scan(delay_s=0.7)
