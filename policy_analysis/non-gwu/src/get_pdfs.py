# Copyright (c) 2025 ph@hallresearch.ai
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

# Requires `sudo apt-get install wkhtmltopdf`

# Python 3.10
# (.venv) patrickh@patrickh-lambda-workstation:~/Workspace/gwsb_caio/policy_analysis/non-gwu$ 
# /home/patrickh/Workspace/gwsb_caio/.venv/bin/python 
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/get_pdfs.py

### imports

from logging_utils import get_logger
logger = get_logger(__name__)

import os
import pdfkit
import re
from timeout_decorator import timeout, TimeoutError
from urllib.parse import urlparse

### establish i/o locations ###################################################

out_dir = f'dat{os.sep}pdf'

### utility function for fetching pdfs from urls ##############################

@timeout(30) # 30 s timeout
def fetch_html2pdf(url, save_dir, file_name, logger=logger, min_kb=2):

    min_size_bytes = min_kb * 1024 # files below 2kb tend to be invalid
    
    pdf_path = os.path.join(save_dir, file_name)

    if os.path.exists(pdf_path):

      return True, pdf_path

    else:

      try:

        pdfkit.from_url(url, pdf_path)

        if os.path.getsize(pdf_path) >= min_size_bytes: 
            logger.info(f'PDF saved to {pdf_path}')
            return True, pdf_path

        else: 

            os.remove(pdf_path)
            logger.info('URL failed to fetch: {url}.')
            return False, None    
        
      except (TimeoutError, Exception) as e:

        logger.error(f'Failed to fetch PDF from {url}. Error: {e}')

        if os.path.exists(pdf_path):
          os.remove(pdf_path)

        return False, None

### utility function for converting urls to filename prefixes #################

def url_to_filename_prefix(url, max_len=80):

    parsed = urlparse(url)

    # host + path (ignore scheme, params, fragment)
    base = parsed.netloc + parsed.path

    # remove leading/trailing slashes
    base = base.strip("/")

    # replace unsafe characters with _
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', base)

    # collapse multiple underscores
    base = re.sub(r'_+', '_', base)

    # trim length
    return base[:max_len]

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
    'https://teaching.ucla.edu/resources/teaching-guides/using-generative-ai-reflectively-and-responsibly-in-teaching-and-learning/'
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
    logger.info(f'Parsing {url} ...')

    title = url_to_filename_prefix(url)

    rc, path = fetch_html2pdf(url, out_dir, title + '.pdf')
        
    if rc:
        logger.info(f'Fetched {url} to {out_dir}{os.sep}{title}.pdf.')
