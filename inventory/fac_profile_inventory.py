import re
import time
import csv
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

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

def run_scan(output_csv="gwsb_faculty_ai_mentions.csv", max_profiles=None, delay_s=0.5, headers=None):
    if headers is None:
        # Fallback to global HEADERS if not explicitly provided (though we'll pass it)
        headers = globals().get('HEADERS', {})
    with requests.Session() as session:
        dir_html = fetch(DIRECTORY_URL, session, headers) # Pass headers to fetch
        profile_urls = extract_profile_links(dir_html)

        if max_profiles:
            profile_urls = profile_urls[:max_profiles]

        rows = []
        for i, url in enumerate(profile_urls, 1):
            try:
                html = fetch(url, session, headers)
                soup = BeautifulSoup(html, "html.parser")
                text = get_visible_text_chunks(soup)

                print(f"[{i}/{len(profile_urls)}] {url} -> {text}")

                hits = find_hits(text)

                rows.append({
                    "profile_url": url,
                    "num_hits": len(hits),
                    "matches": "; ".join(sorted(set(h[0] for h in hits))) if hits else "",
                    "snippets": " || ".join(h[1] for h in hits[:5])  # cap snippets to keep CSV manageable
                })
                print(f"[{i}/{len(profile_urls)}] {url} -> {len(hits)} hits")

            except Exception as e:
                rows.append({
                    "profile_url": url,
                    "num_hits": -1,
                    "matches": "",
                    "snippets": f"ERROR: {e}"
                })
                print(f"[{i}/{len(profile_urls)}] {url} -> ERROR: {e}")

            time.sleep(delay_s)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["profile_url", "num_hits", "matches", "snippets"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved: {output_csv}")

if __name__ == "__main__":
    # Start with a small run to validate selectors, then remove max_profiles.
    run_scan(max_profiles=25, delay_s=0.7)