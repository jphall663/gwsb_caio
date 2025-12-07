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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/gwu/src/txt2chunk.py

### imports and configs
import config 

from logging_utils import get_logger
logger = get_logger(__name__)

import os
import pandas as pd
import re
import tiktoken

MODEL = config.REMOTE_MODEL
LENGTH = config.CHUNK_LENGTH
OVERLAP = 16
MIN_TOKEN_LEN = 1
MAX_TOKEN_LEN = 24

encoding = tiktoken.encoding_for_model(MODEL)

### utility functions #########################################################

def is_non_alpha(line_):

    total_chars = len(line_)
    non_alpha_chars = len(re.findall(r'[^a-zA-Z]', line))

    return non_alpha_chars > (total_chars/2)

def estimate_word_count(file_):

    with open(file_, 'r', encoding='utf-8') as f:
        text = f.read()
        words = text.split()  # splits on any whitespace (spaces, newlines, tabs)
        return len(words)

### establish i/o locations ###################################################

dat_dir = f'dat{os.sep}txt'
out_dir = f'dat{os.sep}chunk'

### loop through text files and chunk them ####################################

cols = ['Type', 'ID', 'Text']

for file in os.listdir(dat_dir):

    logger.info('----------- -----------')
    logger.info(f'Parsing {file} ...')
    [stem, ext] = os.path.splitext(file)
    in_file = dat_dir + os.sep + file

    chunks = pd.DataFrame(columns=cols)
    cache = []  # init cache for storing tokens
    cache_count = 1
    word_count = estimate_word_count(in_file)

    with open(in_file, 'r') as f:

        for line in f:

            if re.match(r'^\s*$', line): # skip blank lines
                continue

            if is_non_alpha(line): # skip lines of numbers or symbols
                continue

            line = line.rstrip()
            line = line.lower()
            line = re.sub(r'http\:\/\/[^\s]*?(\s|$)', r' ', line, flags=re.I) # remove urls
            line = re.sub(r'https\:\/\/[^\s]*?(\s|$)', r' ', line, flags=re.I) # remove urls

            tokens = re.split(r'\s', line)

            for token in tokens:  # cycle through tokens in line

                token = re.sub(r"'", '', token, flags=re.I)

                if token not in ['', ' ', 's'] and\
                    not token.startswith('www') and \
                        MIN_TOKEN_LEN < len(token) < MAX_TOKEN_LEN:

                    if len(cache) <= LENGTH:  # accumulate tokens
                        cache.append(token)
                    else:  # save tokens
                        if not ''.join(cache).isspace():
                            vals = list([stem, cache_count, ' '.join(cache)])
                            row_dict = dict(zip(cols, vals))
                            chunks = pd.concat([chunks, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
                            cache_count += 1
                            cache = cache[-OVERLAP:]
                            cache.append(token)

        else:  # last line of file
            if not ''.join(cache).isspace():
                vals = list([stem, cache_count, ' '.join(cache)])
                row_dict = dict(zip(cols, vals))
                cache_count += 1
                chunks = pd.concat([chunks, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

        logger.info(f'{file} processed into {cache_count} chunks of {LENGTH} tokens.')
        logger.info(f'Approximate token count = {str(cache_count * (LENGTH - OVERLAP))}.')
        logger.info(f'Approximate word count = {word_count}.')

        out_fname = out_dir + os.sep + stem + '.csv'
        chunks.to_csv(out_fname, index=False)

        logger.info(f'Saved: {out_fname}.')
