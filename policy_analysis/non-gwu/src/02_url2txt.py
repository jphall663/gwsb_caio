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

# Python 3.10
# (.venv) patrickh@patrickh-lambda-workstation:~/Workspace/gwsb_caio/policy_analysis/non-gwu$ 
# /home/patrickh/Workspace/gwsb_caio/.venv/bin/python 
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/02_url2txt.py

### imports and config ########################################################

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

from shared.logging_utils import get_logger
logger = get_logger(__name__)

import os
import re
import requests
from urllib.parse import urlparse
from html import unescape
from bs4 import BeautifulSoup  

# Config
TIMEOUT = 30  # seconds
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; policy-fetcher/1.0)'
}

### establish i/o locations ###################################################

BASE_DIR = Path(__file__).resolve().parent.parent
out_dir = BASE_DIR / 'dat' / 'txt'

### utility functions #########################################################

def url_to_filename_prefix(url: str, max_len: int = 80) -> str:
    """
    Convert a URL to a safe filename prefix.
    Mirrors the naming approach used in get_pdfs.py.
    """
    parsed = urlparse(url)
    base = parsed.netloc + parsed.path
    base = base.strip("/")
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', base)
    base = re.sub(r'_+', '_', base)
    return base[:max_len]


def html_to_text(html: str) -> str:
    """
    Convert HTML to readable plain text.
    Uses BeautifulSoup if available; otherwise a minimal fallback.
    """
    if BeautifulSoup:
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(['script', 'style', 'noscript']):
            script.decompose()
        text = soup.get_text(separator=' ')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Fallback: strip tags crudely
    text = re.sub(r'<script.*?>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_text(url: str) -> str | None:
    """
    Fetch the URL and return plain text, or None on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.error(f'Failed to fetch {url}: {exc}')
        return None

    return html_to_text(resp.text)

### cycle through urls to create pdf files ####################################

url_list = [
    'https://students.business.columbia.edu/office-of-student-affairs/academic-advising-and-student-success/academic-integrity/generative-ai-policy',
    'https://ctl.columbia.edu/resources-and-technology/resources/ai-tools/',
    'https://provost.columbia.edu/content/office-senior-vice-provost/ai-policy',
    'https://cndls.georgetown.edu/resources/syllabus-policies/ai-and-homework-support/',
    'https://guides.library.georgetown.edu/ai',
    'https://cndls.georgetown.edu/resources/ai/',
    'https://www.hbs.edu/mba/handbook/standards-of-conduct/academic/Pages/chatgpt-and-ai.aspx',
    'https://registrar.gse.harvard.edu/AI-policy',
    'https://oue.fas.harvard.edu/ai-guidance',
    'https://provost.harvard.edu/guidelines-using-chatgpt-and-other-generative-ai-tools-harvard',
    'https://ist.mit.edu/ai-guidance',
    'https://tll.mit.edu/teaching-resources/course-design/gen-ai-your-course/',
    'https://tlhub.stanford.edu/docs/course-policies-on-generative-ai-use/',
    'https://teachingcommons.stanford.edu/teaching-guides/artificial-intelligence-teaching-guide',
    'https://teachingcommons.stanford.edu/teaching-guides/artificial-intelligence-teaching-guide/creating-your-course-policy-ai',
    'https://communitystandards.stanford.edu/generative-ai-policy-guidance',
    'https://uit.stanford.edu/security/responsibleai',
    'https://ai.universityofcalifornia.edu/governance-transparency/',
    'https://ai.universityofcalifornia.edu/governance-transparency/applicable-law-and-policy.html',
    'https://www.ucop.edu/ethics-compliance-audit-services/_files/compliance/ai/ai-alert.pdf',
    'https://technology.berkeley.edu/AI',
    'https://ethics.berkeley.edu/privacy/appropriate-use-generative-ai-tools',
    'https://dtei.uci.edu/generative-ai/',
    'https://aisc.uci.edu/resources/Statement%20on%20Turnitin%20AI%20detection.pdf',
    'https://guides.library.ucla.edu/c.php?g=1308287&p=9702196',
    'https://online.ucla.edu/chatgpt-and-ai-resources/',
    'https://genai.ucla.edu/',
    'https://genai.ucla.edu/guiding-principles-responsible-use',
    'https://teaching.ucla.edu/resources/teaching-guides/using-generative-ai-reflectively-and-responsibly-in-teaching-and-learning/',
    'https://senate.ucla.edu/news/teaching-guidance-chatgpt-and-related-ai-developments',
    'https://honorcode.nd.edu/ai-recommendations-for-instructors/',
    'https://ai.nd.edu/policies-and-guidelines/',
    'https://honorcode.nd.edu/generative-ai-policy-for-students-august-2023/',
    'https://libguides.usc.edu/generative-AI/home',
    'https://teaching.washington.edu/course-design/ai/',
    'https://teaching.washington.edu/course-design/ai/sample-ai-syllabus-statements/',
    'https://ai.yale.edu/',
    'https://poorvucenter.yale.edu/AIguidance',
    'https://yaledata.yale.edu/yale-university-ai-guidelines-staff',
    'https://provost.yale.edu/news/guidelines-use-generative-ai-tools',
    'https://oercs.berkeley.edu/appropriate-use-generative-ai-tools',
    'https://oercs.berkeley.edu/CERC-AIR'
]

for url in url_list:
    logger.info('----------- -----------')
    logger.info(f'Fetching {url} ...')

    text = fetch_text(url)
    if not text:
        logger.error(f'Skipping {url} (no text extracted).')
        continue

    fname = url_to_filename_prefix(url) + '.txt'
    out_path = out_dir / fname

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    preview = text[:120].replace('\n', ' ')
    logger.info(f'Wrote: {out_path}')
    logger.info(f'Preview: {preview} ...')
