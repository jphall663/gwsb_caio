### imports and configs #######################################################

import csv
import time
from pathlib import Path

from logging_utils import configure_logging, get_logger
from llms.openai_client import gpt_complete

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
