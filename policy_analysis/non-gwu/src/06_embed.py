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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/06_embed.py

### imports and configs #######################################################

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

import os

from shared import config
config.API_KEY = os.environ['OPENAI_API_KEY']

from shared.logging_utils import get_logger
logger = get_logger(__name__)

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from shared.llms import openai_client

embedding_p = config.EMBEDDING_P
BASE_DIR = Path(__file__).resolve().parent.parent

### load data #################################################################

tic = time.time()

data_fname = BASE_DIR / 'dat' / 'nongwu_policy_keyword.csv'
logger.info(f'Loading data file: {data_fname} ...')
data = pd.read_csv(data_fname)
logger.info(data.head())
N = data.shape[0]
logger.info(f'N = {N}')
logger.info(f'Columns = {data.columns}')

### embed text ################################################################

logger.info(f'Initializing embeddings for {N} rows ...')

embedding_names = ['dim_' + str(i) for i in range(0, embedding_p)]

for i in tqdm(range(0, N)):

    # for saving each iteration in case of API crash
    output_fname = BASE_DIR / 'dat' / 'nongwu_policy_keyword_embed.csv'
    if output_fname.exists():
        data_ = pd.read_csv(output_fname, low_memory=True)
    else:
        data_ = data

    # extract text and embed
    content = data_.loc[i, 'Keywords']
    if content is not np.nan:
        row_embedding = np.array(openai_client.gpt_embed(content)).reshape(1, embedding_p)
        data_.loc[i, embedding_names] = row_embedding[0]
    else:
        data_.loc[i, embedding_names] = np.zeros((1, embedding_p))

    # for saving each iteration in case of API crash
    data_.to_csv(output_fname, index=False)

### verify putput ###########################################################

logger.info(f'Correct data shape: {data_.shape == (N, 1539)}.')

no_empties = data_.isna().any().any()
logger.info(f'Any empty data cells: {no_empties}.')

# end timer
toc = time.time() - tic
logger.info(f'All tasks performed in {toc:.2f} s.')
