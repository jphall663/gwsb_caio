###############################################################################
#                                                                             #
# Screen personal websites in gwsb_faculty_ai_mentions.csv before running !!! #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from logging_utils import logging, configure_logging, get_logger

### constants #################################################################

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

AI_PATTERNS = [
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

AI_REGEX = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
CV_DIR = BASE_DIR / "cv"

DEFAULT_INPUT_CSV = DATA_DIR / "gwsb_faculty_ai_mentions.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "per_site_ai_mentions.csv"

configure_logging()
logger = get_logger(__name__)

### utilities #################################################################

def fetch(url, session, headers, timeout=30):
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def get_visible_text_chunks(soup):
    main = soup.find("main") or soup.body or soup
    for tag in main.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return main.get_text("\n", strip=True)

def find_hits(text, window=120):
    hits = []
    for m in AI_REGEX.finditer(text):
        start = max(m.start() - window, 0)
        end = min(m.end() + window, len(text))
        snippet = text[start:end].replace("\n", " ")
        hits.append((m.group(0), snippet))
    return hits

def download_cv_if_present(soup, session, headers, base_url):
    """
    Locate a CV link on a personal site, download it, and return the saved filename.
    """
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cv_keywords = ("cv", "c.v.", "curriculum vitae", "vitae")

    def _download_from_anchor(a):
        href = a.get("href", "")
        if href.startswith(("mailto:", "tel:", "#", "javascript:")):
            return ""

        cv_url = urljoin(base_url, href)
        try:
            resp = session.get(cv_url, headers=headers, timeout=30)
            resp.raise_for_status()
            filename = Path(urlparse(cv_url).path).name or "cv_download"
            dest = CV_DIR / filename
            if dest.exists():
                dest = CV_DIR / f"{dest.stem}_{int(time.time())}{dest.suffix}"

            with open(dest, "wb") as f:
                f.write(resp.content)

            logger.info("Downloaded CV to %s", dest)
            return dest.name
        except Exception as e:
            logger.warning("Failed to download CV from %s: %s", cv_url, e)
            return ""

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        if any(kw in text for kw in cv_keywords):
            saved = _download_from_anchor(a)
            if saved:
                return saved

    return ""

def parse_personal_urls(value):
    if not value:
        return []
    urls = [u.strip() for u in value.split(";") if u.strip()]
    return urls

### main entrypoint ###########################################################

def run_scan(input_csv=DEFAULT_INPUT_CSV, output_csv=DEFAULT_OUTPUT_CSV, delay_s=0.5, headers=None):
    if headers is None:
        headers = globals().get("HEADERS", {})

    rows = []
    with requests.Session() as session:
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                existing_cv = (row.get("cv_filename") or "").strip()
                personal_urls = parse_personal_urls(row.get("personal_urls", ""))
                for personal_url in personal_urls:
                    try:
                        html = fetch(personal_url, session, headers)
                        soup = BeautifulSoup(html, "html.parser")
                        text = get_visible_text_chunks(soup)
                        hits = find_hits(text)
                        snippets = " || ".join(h[1] for h in hits[:5])
                        cv_filename = ""
                        if not existing_cv:
                            cv_filename = download_cv_if_present(soup, session, headers, personal_url)

                        rows.append({
                            "personal_url": personal_url,
                            "name": name,
                            "num_hits": len(hits),
                            "matches": "; ".join(sorted(set(h[0] for h in hits))) if hits else "",
                            "snippets": snippets,
                            "cv_filename": cv_filename
                        })
                        logger.info("%s -> %d hits", personal_url, len(hits))
                    except Exception as e:
                        rows.append({
                            "personal_url": personal_url,
                            "name": name,
                            "num_hits": -1,
                            "matches": "",
                            "snippets": f"ERROR: {e}",
                            "cv_filename": ""
                        })
                        logger.error("%s -> ERROR: %s", personal_url, e)

                    time.sleep(delay_s)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["personal_url", "name", "num_hits", "matches", "snippets", "cv_filename"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    run_scan(delay_s=0.7)
