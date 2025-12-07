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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/gwu/src/concat_csv.py

### imports

from logging_utils import get_logger
logger = get_logger(__name__)

from pathlib import Path
import pandas as pd
import os

### stack input txt files into a dataframe ####################################

in_dir = Path(f'dat{os.sep}chunk')
out_csv = Path(f'dat{os.sep}chunk{os.sep}existing_policy_combined.csv')

files = sorted(in_dir.glob('*.csv'))
if not files:
    raise FileNotFoundError(f'No CSV files found in {in_dir}')

dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(out_csv, index=False)

logger.info(f'Combined {len(files)} files -> {out_csv}')
logger.info(f'Total rows: {len(combined):,}')