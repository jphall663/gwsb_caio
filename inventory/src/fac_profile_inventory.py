### imports and configs #######################################################

import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from logging_utils import logging, configure_logging, get_logger

import requests
from bs4 import BeautifulSoup

### constants #################################################################

BASE = "https://business.gwu.edu"
DIRECTORY_URL = "https://business.gwu.edu/faculty-directory"

# Expand/adjust this list as you like.
AI_PATTERNS = [
    r"\bAI\b",
    r"artificial intelligence",
    r"machine learning",
    r"deep learning",
    r"neural network",
    r"natural language processing|\bNLP\b",
    r"large language model|\bLLM\b",
    r"generative AI|genAI",
    r"ChatGPT",
    r"automation",
    r"algorithmic|algorithms?",
    r"data science",
    r"predictive modeling|prediction",
    r"computer vision",
    r"reinforcement learning",
]

AI_REGEX = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)
PROFILE_PATTERN = re.compile(r"^/[a-z0-9\-]+/?$")
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "strong")
CONTACT_LABEL_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "strong", "div", "span", "p")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
CV_DIR = BASE_DIR / "cv"

configure_logging()
logger = get_logger(__name__)

### scrape utilities ##########################################################

def fetch(url, session, headers, timeout=30):
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def normalize_profile_url(href):
    if not href:
        return None

    # Skip mail, tel, anchors, JS
    if href.startswith(("mailto:", "tel:", "#", "javascript:")):
        return None

    abs_url = urljoin(BASE, href)
    parsed = urlparse(abs_url)

    # Must be business.gwu.edu only
    if parsed.netloc != urlparse(BASE).netloc:
        return None

    # Must match profile slug pattern
    if not PROFILE_PATTERN.match(parsed.path):
        return None

    return abs_url.rstrip("/")

def extract_profile_links(directory_html):
    soup = BeautifulSoup(directory_html, "html.parser")
    links = set()

    # Conservative approach:
    # gather links within main content area; if site structure changes,
    # you can broaden this selector.
    main = soup.find("main") or soup
    for a in main.find_all("a", href=True):
        href = a["href"]
        url = normalize_profile_url(href)
        if not url:
            continue
        # Light heuristic: skip obvious navigation or directory self-links
        if url.rstrip("/") == DIRECTORY_URL.rstrip("/"):
            continue
        # Skip common non-profile paths (tune if needed)
        if any(seg in url for seg in ["/academics/", "/admissions/", "/news", "/events", "/about", "/research", "/staff-directory"]):
            continue
        links.add(url)

    return sorted(links)

def get_visible_text_chunks(soup):
    """
    Extracts text from what is likely the page's main content.
    Also tries to preserve "tab sections" by splitting on headings.
    """
    main = soup.find("main") or soup.body or soup

    # Remove nav/footer/script/style to reduce noise
    for tag in main.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    text = main.get_text("\n", strip=True)

    return text

def find_hits(text, window=120):
    hits = []
    for m in AI_REGEX.finditer(text):
        start = max(m.start() - window, 0)
        end = min(m.end() + window, len(text))
        snippet = text[start:end].replace("\n", " ")
        hits.append((m.group(0), snippet))
    return hits

def iter_contact_links(soup):
    """
    Yield anchor tags that appear under headings containing 'Contact'.
    """
    for block in iter_contact_sections(soup):
        for a in block.find_all("a", href=True):
            yield a

def iter_contact_sections(soup):
    """
    Yield tag blocks under any heading/label containing 'Contact'.
    """
    root = soup.find("main") or soup
    for tag in root.find_all(CONTACT_LABEL_TAGS):
        if "contact" not in tag.get_text(" ", strip=True).lower():
            continue
        if tag.find_parent(["nav", "header", "footer"]):
            continue  # avoid global nav/footer "Contact" links

        yield tag.parent or tag

        for sib in tag.find_next_siblings():
            sib_text = sib.get_text(" ", strip=True).lower()
            if sib.name in CONTACT_LABEL_TAGS and "contact" in sib_text:
                break
            if sib.name in HEADING_TAGS:
                break
            yield sib

def extract_personal_urls(soup):
    """
    Pull likely personal/contact URLs from a profile page.
    - Anchors in Contact blocks (preferring a single external/personal link).
    """
    urls = set()
    website_keywords = ("personal website", "website", "web site", "homepage", "home page", "personal page", "portfolio")
    contact_blocklist = {
        "business.gwu.edu",
        "www.gwu.edu",
        "gwu.edu",
        "accessibility.gwu.edu",
        "campusadvisories.gwu.edu",
        "compliance.gwu.edu",
        "privacy.gwu.edu",
        "research.gwu.edu",
        "library.gwu.edu",
        "online.business.gwu.edu",
        "give.gwu.edu",
        "forms.gle",
        "google.com",
        "facebook.com",
        "www.facebook.com",
        "twitter.com",
        "x.com",
        "www.instagram.com",
        "instagram.com",
        "www.linkedin.com",
        "linkedin.com",
        "www.youtube.com",
        "youtube.com",
        "sciencedirect.com",
        "www.sciencedirect.com",
        "research.stlouisfed.org",
        "doi.org",
        "dx.doi.org",
        "scholar.google.com",
        "ssrn.com",
        "papers.ssrn.com",
        "arxiv.org",
        "researchgate.net",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "elsevier.com",
        "www.sciencedirect.com",
        "sagepub.com",
        "journals.sagepub.com",
        "tandfonline.com",
        "www.tandfonline.com",
        "springer.com",
        "link.springer.com",
        "academic.oup.com",
        "cambridge.org",
        "iop.org",
        "ieee.org",
        "dl.acm.org",
    }

    def _maybe_add(a, allow_external: bool):
        href = a.get("href", "")
        if href.startswith(("mailto:", "tel:", "#", "javascript:")):
            return
        text = a.get_text(" ", strip=True).lower()
        url = urljoin(BASE, href)

        parsed = urlparse(url)
        base_host = urlparse(BASE).netloc
        if url.lower().endswith(".pdf"):
            return  # avoid CVs and document links

        mentions_site = any(k in text for k in website_keywords)

        if parsed.netloc == base_host:
            if mentions_site:
                urls.add(url)
            return

        # External link: only accept if allowed or explicitly labeled as a website/homepage.
        host = parsed.netloc.lower()
        if host in contact_blocklist:
            return

        if allow_external and parsed.netloc and parsed.netloc != base_host:
            urls.add(url)
        elif mentions_site:
            urls.add(url)

    # Contact section anchors: allow external links, even if not keyworded.
    contact_links = list(iter_contact_links(soup))
    for a in contact_links:
        _maybe_add(a, allow_external=True)
        if len(urls) >= 1:
            break  # prefer the first suitable contact link

    # Filter out obvious non-links (mailto/tel/anchors/js)
    cleaned = []
    for url in urls:
        if url.startswith(("mailto:", "tel:", "#", "javascript:")):
            continue
        cleaned.append(url)

    return sorted(cleaned)

def download_cv_if_present(soup, session, headers):
    """
    Locate a CV link in the Contact section, download it, and return the saved filename.
    """
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cv_keywords = ("cv", "c.v.", "curriculum vitae", "vitae")

    def _download_from_anchor(a):
        href = a.get("href", "")
        if href.startswith(("mailto:", "tel:", "#", "javascript:")):
            return ""

        cv_url = urljoin(BASE, href)
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

    # First try anchors in contact sections
    for a in iter_contact_links(soup):
        text = a.get_text(" ", strip=True).lower()
        if any(kw in text for kw in cv_keywords):
            saved = _download_from_anchor(a)
            if saved:
                return saved

    # Fallback: any anchor on the page mentioning CV keywords
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        if any(kw in text for kw in cv_keywords):
            saved = _download_from_anchor(a)
            if saved:
                return saved

    return ""

def extract_faculty_name(soup):
    """
    Heuristic: use first <h1> text as the faculty name, fallback to first <h2>.
    """
    for tag in ("h1", "h2"):
        h = soup.find(tag)
        if h:
            name = h.get_text(strip=True)
            if name:
                return name
    return ""

### main entrypoint ###########################################################

DEFAULT_OUTPUT_CSV = DATA_DIR / "gwsb_faculty_ai_mentions.csv"

def run_scan(output_csv=DEFAULT_OUTPUT_CSV, max_profiles=None, delay_s=0.5, headers=None):
    if headers is None:
        # Fallback to global HEADERS if not explicitly provided (though we'll pass it)
        headers = globals().get('HEADERS', {})
    with requests.Session() as session:
        dir_html = fetch(DIRECTORY_URL, session, headers) # Pass headers to fetch
        profile_urls = extract_profile_links(dir_html)

        if max_profiles:
            profile_urls = profile_urls[:max_profiles]

        total = len(profile_urls)
        logger.info("Scanning %d profiles from %s", total, DIRECTORY_URL)

        rows = []
        for i, url in enumerate(profile_urls, 1):
            try:
                html = fetch(url, session, headers)
                soup = BeautifulSoup(html, "html.parser")
                text = get_visible_text_chunks(soup)

                logger.debug("[%d/%d] %s -> %s", i, total, url, text)

                hits = find_hits(text)
                personal_urls = extract_personal_urls(soup)
                cv_filename = download_cv_if_present(soup, session, headers)
                faculty_name = extract_faculty_name(soup)

                rows.append({
                    "profile_url": url,
                    "name": faculty_name,
                    "num_hits": len(hits),
                    "matches": "; ".join(sorted(set(h[0] for h in hits))) if hits else "",
                    "snippets": " || ".join(h[1] for h in hits[:5]),  # cap snippets to keep CSV manageable
                    "personal_urls": "; ".join(personal_urls),
                    "cv_filename": cv_filename
                })
                logger.info("[%d/%d] %s -> %d hits", i, total, url, len(hits))

            except Exception as e:
                rows.append({
                    "profile_url": url,
                    "name": "",
                    "num_hits": -1,
                    "matches": "",
                    "snippets": f"ERROR: {e}",
                    "personal_urls": "",
                    "cv_filename": ""
                })
                logger.error("[%d/%d] %s -> ERROR: %s", i, total, url, e)

            time.sleep(delay_s)

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["profile_url", "name", "num_hits", "matches", "snippets", "personal_urls", "cv_filename"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    # Start with a small run to validate selectors, then remove max_profiles.
    run_scan(delay_s=0.7)
