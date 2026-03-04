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
# Screen personal websites in gwsb_faculty_ai_mentions.csv before running !!! #
#                                                                             #
###############################################################################

### imports and configs #######################################################

import csv
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

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

from shared.logging_utils import logging, configure_logging, get_logger
from ai_terms import get_ai_regex

### constants #################################################################

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
CV_DIR = BASE_DIR / "cv"

DEFAULT_INPUT_CSV = DATA_DIR / "gwsb_faculty_ai_mentions.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "per_site_ai_mentions.csv"

configure_logging(level=logging.DEBUG)
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
    ai_regex = get_ai_regex()
    for m in ai_regex.finditer(text):
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
                        #logger.debug(hits)
                        snippets = " || ".join(h[1] for h in hits)
                        #logger.debug(snippets)
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
