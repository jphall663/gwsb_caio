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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/05_apply_keywords.py

# Requires keywords and outputs from https://github.com/jphall663/nmf

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

from shared.logging_utils import get_logger
logger = get_logger(__name__)

import os
import pandas as pd
from wordcloud import WordCloud

pd.set_option('display.max_rows', None)

BASE_DIR = Path(__file__).resolve().parent.parent

### set keywords ##############################################################

keyword_list = [
    'academic',
    'accessibility',
    'accountability',
    'accuracy',
    'act',
    'adapt',
    'advice',
    'agreement',
    'air',
    'analyze',
    'appendix',
    'applicable',
    'arrow',
    'artificialintelligence',
    'assessment',
    'assignment',
    'bias',
    'billing',
    'blend',
    'business',
    'campus',
    'career',
    'case',
    'cerc',
    'charter',
    'chatbots',
    'chatgpt',
    'class',
    'classroom',
    'committee',
    'community',
    'compliance',
    'computer',
    'concept',
    'conflict',
    'consultation',
    'contract',
    'coursework',
    'create',
    'data',
    'design',
    'development',
    'digital',
    'disclosure',
    'discrimination',
    'disruption',
    'draft',
    'education',
    'educational',
    'engage',
    'engineer',
    'enrollment',
    'enterprise',
    'ethic',
    'ethical',
    'event',
    'expectation',
    'experience',
    'external',
    'faculty',
    'fail',
    'faqs',
    'framework',
    'gai',
    'genai',
    'generative',
    'goal',
    'governance',
    'graduate',
    'guidance',
    'guide',
    'guideline',
    'health',
    'hgse',
    'hipaa',
    'honor',
    'human',
    'hybrid',
    'idea',
    'impact',
    'important',
    'inclusive',
    'initiative',
    'innovation',
    'insight',
    'instructional',
    'instructors',
    'integrity',
    'keyboard',
    'kit',
    'law',
    'learn',
    'level',
    'leverage',
    'library',
    'messaging',
    'news',
    'office',
    'open',
    'openai',
    'opportunity',
    'overview',
    'pattern',
    'personal',
    'policy',
    'practice',
    'privacy',
    'problem',
    'professor',
    'program',
    'project',
    'prompt',
    'proposal',
    'protection',
    'quizzes',
    'record',
    'research',
    'resource',
    'retention',
    'review',
    'risk',
    'sanction',
    'school',
    'science',
    'security',
    'seed',
    'senate',
    'solution',
    'solve',
    'space',
    'statement',
    'strategy',
    'student',
    'syllabus',
    'symposium',
    'system',
    'teach',
    'team',
    'technology',
    'tool',
    'topic',
    'training',
    'transparency',
    'uit',
    'university',
    'usage',
    'workshop',
    'world',
    'write',
    'writers'
]

### load data #################################################################

lemmatized_data_fname = BASE_DIR / 'out' / '_raw_lower_rgx_entity_stemmed_stopped_long_freq0.txt'
lemmatized_data = pd.read_csv(lemmatized_data_fname, header=None, skip_blank_lines=False)
logger.info(f'Loaded: {lemmatized_data_fname}.')
logger.info(lemmatized_data.head())

chunks_fname = BASE_DIR / 'dat' / 'chunk' / 'nongwu_policy_combined.csv'
chunk_data = pd.read_csv(chunks_fname)
logger.info(f'Loaded: {chunks_fname}.')
logger.info(chunk_data.head())

if chunk_data.shape[0] == lemmatized_data.shape[0]:
        logger.info('Loaded sets have same N.')
else:
    logger.error('Loaded sets have different N.')
    logger.info(f'lemmatized_data N: {lemmatized_data.shape[0]}.')
    logger.info(f'chunk_data N: {chunk_data.shape[0]}.')
    sys.exit(-1)

### keyword tagging ###########################################################

big_list = []
chunk_data['Keywords'] = ''
for i in range(0, chunk_data.shape[0]):

    row_string = ''

    for kw in keyword_list:
        if kw in str(lemmatized_data.iloc[i, 0]).split(' '):
            big_list.append(kw)
            row_string += kw
            row_string += ', '

    row_string = row_string[:-2]  # remove trailing comma
    chunk_data.loc[i, 'Keywords'] = row_string
    text = chunk_data.loc[i, 'Text']
    kws = chunk_data.loc[i, 'Keywords']

    if (i + 1) % 100 == 0:
        logger.info('----------- -----------')
        logger.info(f'Row: {str(i + 1)}/{chunk_data.shape[0]}')
        logger.info(f'Chunk text: {text}')
        logger.info(f'Chunk topics: {kws}')

### save output data ##########################################################

chunk_data_fname = BASE_DIR / 'dat' / 'nongwu_policy_keyword.csv'
chunk_data.to_csv(chunk_data_fname, index=False)
logger.info(f'Saved: {chunk_data_fname}.')

### word cloud ################################################################

logger.info('Generating word cloud ...')

wc_text = ' '.join(big_list)

wordcloud = WordCloud(
    width=4000,
    height=2000,
    max_words=100,       # maximum number of words shown
    min_font_size=7,     # minimum font size
    background_color='white',
    colormap='Set2',  # pastel colormap
    collocations=False
).generate(wc_text)

# save to file
wc_fname = f'out{os.sep}res{os.sep}nongwu_policy_key_word_cloud_hi_4k.png'
wordcloud.to_file(wc_fname)
logger.info(f'Saved: {wc_fname}.')
