### imports and configs #######################################################

import csv
from pathlib import Path

from logging_utils import logging, configure_logging, get_logger

### constants #################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
DEFAULT_PROFILE_CSV = DATA_DIR / "gwsb_faculty_ai_mentions.csv"
DEFAULT_PER_SITE_CSV = DATA_DIR / "per_site_ai_mentions.csv"
DEFAULT_CV_CSV = DATA_DIR / "cv_ai_mentions.csv"
DEFAULT_MANUAL_CSV = DATA_DIR / "manual_ai_mentions.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "fac_ai_mentions_joined.csv"

configure_logging()
logger = get_logger(__name__)

### helpers ###################################################################

def parse_hits(value):
    try:
        return int(value)
    except Exception:
        return 0

def normalize_name(name):
    return (name or "").strip().lower()

def add_unique(accum, value):
    value = (value or "").strip()
    if not value:
        return
    parts = [p.strip() for p in value.split(";") if p.strip()]
    for part in parts:
        if part not in accum:
            accum.append(part)

def add_snippets(accum, value, cap=1000):
    value = (value or "").strip()
    if not value:
        return
    parts = [p.strip() for p in value.split("||") if p.strip()]
    for part in parts:
        if part not in accum and len(accum) < cap:
            accum.append(part)

def add_manual_snippets(accum, value):
    value = (value or "").strip()
    if not value:
        return
    accum.append(value)

def ensure_record(store, name):
    key = normalize_name(name)
    if not key:
        return None
    if key not in store:
        store[key] = {
            "name": name.strip(),
            "fac_profile_url": "",
            "fac_profile_num_hits": 0,
            "fac_profile_matches": [],
            "fac_profile_snippets": [],
            "per_site_url": [],
            "per_site_num_hits": 0,
            "per_site_matches": [],
            "per_site_snippets": [],
            "cv_filename": [],
            "cv_num_hits": 0,
            "cv_matches": [],
            "cv_snippets": [],
            "manual_filename": [],
            "manual_num_hits": 0,
            "manual_matches": [],
            "manual_snippets": [],
        }
    return store[key]

### main entrypoint ###########################################################

def run_join(
    profile_csv=DEFAULT_PROFILE_CSV,
    per_site_csv=DEFAULT_PER_SITE_CSV,
    cv_csv=DEFAULT_CV_CSV,
    manual_csv=DEFAULT_MANUAL_CSV,
    output_csv=DEFAULT_OUTPUT_CSV,
):
    records = {}

    if profile_csv.exists():
        with profile_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_hits = parse_hits(row.get("num_hits", ""))
                if num_hits <= 0:
                    continue
                rec = ensure_record(records, row.get("name", ""))
                if not rec:
                    continue
                rec["fac_profile_url"] = (row.get("profile_url") or "").strip()
                rec["fac_profile_num_hits"] += num_hits
                add_unique(rec["fac_profile_matches"], row.get("matches", ""))
                add_snippets(rec["fac_profile_snippets"], row.get("snippets", ""))
    else:
        logger.warning("Missing input CSV: %s", profile_csv)

    if per_site_csv.exists():
        with per_site_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_hits = parse_hits(row.get("num_hits", ""))
                if num_hits <= 0:
                    continue
                rec = ensure_record(records, row.get("name", ""))
                if not rec:
                    continue
                add_unique(rec["per_site_url"], row.get("personal_url", ""))
                rec["per_site_num_hits"] += num_hits
                add_unique(rec["per_site_matches"], row.get("matches", ""))
                add_snippets(rec["per_site_snippets"], row.get("snippets", ""))
    else:
        logger.warning("Missing input CSV: %s", per_site_csv)

    if cv_csv.exists():
        with cv_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_hits = parse_hits(row.get("num_hits", ""))
                if num_hits <= 0:
                    continue
                rec = ensure_record(records, row.get("name", ""))
                if not rec:
                    continue
                add_unique(rec["cv_filename"], row.get("cv_filename", ""))
                rec["cv_num_hits"] += num_hits
                add_unique(rec["cv_matches"], row.get("matches", ""))
                add_snippets(rec["cv_snippets"], row.get("snippets", ""))
    else:
        logger.warning("Missing input CSV: %s", cv_csv)

    if manual_csv.exists():
        with manual_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_hits = parse_hits(row.get("num_hits", ""))
                if num_hits <= 0:
                    continue
                rec = ensure_record(records, row.get("name", ""))
                if not rec:
                    continue
                add_unique(rec["manual_filename"], row.get("filename", ""))
                rec["manual_num_hits"] += num_hits
                add_unique(rec["manual_matches"], row.get("matches", ""))
                add_manual_snippets(rec["manual_snippets"], row.get("snippets", ""))
    else:
        logger.warning("Missing input CSV: %s", manual_csv)

    rows = []
    for key in sorted(records):
        rec = records[key]
        total_hits = (
            rec["fac_profile_num_hits"]
            + rec["per_site_num_hits"]
            + rec["cv_num_hits"]
            + rec["manual_num_hits"]
        )
        rows.append({
            "name": rec["name"],
            "fac_profile_url": rec["fac_profile_url"],
            "fac_profile_num_hits": rec["fac_profile_num_hits"],
            "fac_profile_matches": "; ".join(rec["fac_profile_matches"]),
            "fac_profile_snippets": " || ".join(rec["fac_profile_snippets"]),
            "per_site_url": "; ".join(rec["per_site_url"]),
            "per_site_num_hits": rec["per_site_num_hits"],
            "per_site_matches": "; ".join(rec["per_site_matches"]),
            "per_site_snippets": " || ".join(rec["per_site_snippets"]),
            "cv_filename": "; ".join(rec["cv_filename"]),
            "cv_num_hits": rec["cv_num_hits"],
            "cv_matches": "; ".join(rec["cv_matches"]),
            "cv_snippets": " || ".join(rec["cv_snippets"]),
            "manual_filename": "; ".join(rec["manual_filename"]),
            "manual_num_hits": rec["manual_num_hits"],
            "manual_matches": "; ".join(sorted(set(rec["manual_matches"]))),
            "manual_snippets": " ".join(rec["manual_snippets"]),
            "total_hits": total_hits,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
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
                "manual_filename",
                "manual_num_hits",
                "manual_matches",
                "manual_snippets",
                "total_hits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    run_join()
