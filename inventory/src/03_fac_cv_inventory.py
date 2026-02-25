### imports and configs #######################################################

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

from logging_utils import logging, configure_logging, get_logger
from ai_terms import get_ai_regex

from bs4 import BeautifulSoup

### constants #################################################################

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dat"
CV_DIR = BASE_DIR / "cv"
DEFAULT_INPUT_CSVS = [
    DATA_DIR / "gwsb_faculty_ai_mentions.csv",
    DATA_DIR / "per_site_ai_mentions.csv",
]
DEFAULT_OUTPUT_CSV = DATA_DIR / "cv_ai_mentions.csv"

configure_logging()
logger = get_logger(__name__)

### parsing utilities #########################################################

def _read_pdf_text(path: Path) -> str:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        PyPDF2 = None

    if PyPDF2:
        text_parts = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(str(path))
    except Exception as e:
        raise RuntimeError(f"No PDF parser available ({e})")

def _read_docx_text(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception as e:
        raise RuntimeError(f"No DOCX parser available ({e})")

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def _read_doc_text(path: Path) -> str:
    try:
        import textract  # type: ignore
    except Exception as e:
        raise RuntimeError(f"No DOC parser available ({e})")

    data = textract.process(str(path))
    return data.decode("utf-8", errors="ignore")

def _read_html_text(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext == ".docx":
        return _read_docx_text(path)
    if ext == ".doc":
        return _read_doc_text(path)
    if ext in {".html", ".htm"}:
        return _read_html_text(path)
    raise RuntimeError(f"Unsupported file type: {ext}")

def find_hits(text: str, window: int = 120):
    hits = []
    ai_regex = get_ai_regex()
    for m in ai_regex.finditer(text):
        start = max(m.start() - window, 0)
        end = min(m.end() + window, len(text))
        snippet = text[start:end].replace("\n", " ")
        hits.append((m.group(0), snippet))
    return hits

### csv utilities #############################################################

def load_cv_map(paths: Iterable[Path]) -> Dict[str, str]:
    cv_map: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            logger.warning("Input CSV not found: %s", path)
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("cv_filename") or "").strip()
                if not filename:
                    continue
                name = (row.get("name") or "").strip()
                if filename not in cv_map or (not cv_map[filename] and name):
                    cv_map[filename] = name
    return cv_map

### main entrypoint ###########################################################

def run_scan(input_csvs=DEFAULT_INPUT_CSVS, output_csv=DEFAULT_OUTPUT_CSV):
    cv_map = load_cv_map(input_csvs)
    rows = []

    for filename, name in sorted(cv_map.items()):
        path = CV_DIR / filename
        if not path.exists():
            logger.error("Missing CV file: %s", path)
            rows.append({
                "cv_filename": filename,
                "name": name,
                "num_hits": "",
                "matches": "",
                "snippets": ""
            })
            continue

        try:
            text = extract_text(path)
            hits = find_hits(text)
            rows.append({
                "cv_filename": filename,
                "name": name,
                "num_hits": len(hits),
                "matches": "; ".join(sorted(set(h[0] for h in hits))) if hits else "",
                "snippets": " || ".join(h[1] for h in hits),
            })
            logger.info("%s -> %d hits", filename, len(hits))
        except Exception as e:
            logger.error("%s -> ERROR: %s", filename, e)
            rows.append({
                "cv_filename": filename,
                "name": name,
                "num_hits": "",
                "matches": "",
                "snippets": ""
            })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cv_filename", "name", "num_hits", "matches", "snippets"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", output_csv)

if __name__ == "__main__":
    run_scan()
