###############################################################################
#                                                                             #
# Maintain dat/manual_entries.csv before running !!!                          #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import csv
from pathlib import Path

from logging_utils import logging, configure_logging, get_logger
from ai_terms import get_ai_regex

### constants #################################################################
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
DEFAULT_INPUT_CSV = DATA_DIR / "manual_entries.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "manual_ai_mentions.csv"

configure_logging()
logger = get_logger(__name__)

### utilities #################################################################

def find_hits(text, window=120):
    hits = []
    ai_regex = get_ai_regex()
    for m in ai_regex.finditer(text):
        start = max(m.start() - window, 0)
        end = min(m.end() + window, len(text))
        snippet = text[start:end].replace("\n", " ")
        hits.append((m.group(0), snippet))
    return hits

### main entrypoint ###########################################################

def run_scan(input_csv=DEFAULT_INPUT_CSV, output_csv=DEFAULT_OUTPUT_CSV):
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    records = {}
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Name") or row.get("name") or "").strip()
            item = (row.get("Item") or row.get("item") or "").strip()
            if not name and not item:
                continue

            try:
                hits = find_hits(item)
                key = name.lower()
                if key not in records:
                    records[key] = {
                        "filename": input_csv.name,
                        "name": name,
                        "num_hits": 0,
                        "matches": set(),
                        "snippets": [],
                    }

                records[key]["num_hits"] += len(hits)
                records[key]["matches"].update(h[0] for h in hits)
                records[key]["snippets"].extend(h[1] for h in hits)
                logger.info("%s (%s) -> %d hits", name, input_csv.name, len(hits))
            except Exception as e:
                logger.error("%s (%s) -> ERROR: %s", name, input_csv.name, e)

    rows = []
    for key in sorted(records):
        rec = records[key]
        rows.append({
            "filename": rec["filename"],
            "name": rec["name"],
            "num_hits": rec["num_hits"],
            "matches": "; ".join(sorted(rec["matches"])),
            "snippets": " ".join(rec["snippets"]),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "name", "num_hits", "matches", "snippets"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    run_scan()
