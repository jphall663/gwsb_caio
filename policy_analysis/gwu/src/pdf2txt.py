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

# Python 3.10
# (.venv) patrickh@patrickh-lambda-workstation:~/Workspace/gwsb_caio/policy_analysis/gwu$ 
# /home/patrickh/Workspace/gwsb_caio/.venv/bin/python 
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/gwu/src/pdf2txt.py

### imports

from logging_utils import get_logger
logger = get_logger(__name__)

import os
import tika
from tika import parser
tika.TikaClientOnly = True

### establish i/o locations ###################################################

dat_dir = f'dat{os.sep}pdf'
out_dir = f'dat{os.sep}txt'

### cycle through pdfs to create text files ###################################

for file in os.listdir(dat_dir):

    logger.info('----------- -----------')
    logger.info(f'Parsing {file} ...')
    [stem, ext] = os.path.splitext(file)
    in_file = dat_dir + os.sep + file
    pdf_contents = parser.from_file(in_file, requestOptions={'timeout': 120})
    preview = pdf_contents['content'][:100].replace('\n', ' ').strip()
    logger.info(f'Preview: {preview} ...')

    out_file = out_dir + os.sep + stem + '.txt'

    with open(out_file, 'w') as txt_file:
        txt_file.write(pdf_contents['content'])
        logger.info(f'Wrote: {out_file}.')
