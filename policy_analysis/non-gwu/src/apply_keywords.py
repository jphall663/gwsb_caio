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
# (.venv) patrickh@patrickh-lambda-workstation:~/Workspace/gwsb_caio/policy_analysis/non-gwu$ 
# /home/patrickh/Workspace/gwsb_caio/.venv/bin/python 
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/apply_keywords.py

### imports and configs #######################################################

from logging_utils import get_logger
logger = get_logger(__name__)

import os
import pandas as pd
import sys
from wordcloud import WordCloud

pd.set_option('display.max_rows', None)

### set keywords ##############################################################

keyword_list = [
    'ability',
    'academic',
    'acceptable',
    'accessibility',
    'act',
    'active',
    'activity',
    'adapt',
    'admissions',
    'advice',
    'advisory',
    'agreement',
    'air',
    'alumni',
    'analysis',
    'analyze',
    'apm',
    'app',
    'appendix',
    'applicable',
    'arguments',
    'arrow',
    'artificialintelligence',
    'assess',
    'assessment',
    'assignment',
    'authentic',
    'autumn',
    'aware',
    'awareness',
    'bca',
    'benefit',
    'bias',
    'billing',
    'blend',
    'blog',
    'build',
    'business',
    'campus',
    'canvas',
    'capability',
    'career',
    'case',
    'cash',
    'cerc',
    'certificate',
    'challenge',
    'charter',
    'chat',
    'chatbots',
    'chatgpt',
    'citation',
    'civil',
    'class',
    'classification',
    'classroom',
    'collaboration',
    'committee',
    'community',
    'compliance',
    'computer',
    'concept',
    'conference',
    'confidential',
    'conflict',
    'consideration',
    'consultation',
    'context',
    'contract',
    'copyrighted',
    'council',
    'coursework',
    'create',
    'creative',
    'critical',
    'critically',
    'ctl',
    'culture',
    'data',
    'demonstrate',
    'design',
    'detection',
    'development',
    'device',
    'digital',
    'disability',
    'disclosure',
    'discourse',
    'discrimination',
    'disruption',
    'draft',
    'dtei',
    'edition',
    'education',
    'effective',
    'electronic',
    'engage',
    'engagement',
    'engineer',
    'enrollment',
    'enterprise',
    'ethic',
    'ethical',
    'evaluate',
    'evaluation',
    'event',
    'evolve',
    'exam',
    'excellence',
    'expectation',
    'experience',
    'expert',
    'explore',
    'external',
    'faculty',
    'fail',
    'faqs',
    'federal',
    'feel',
    'fellows',
    'focus',
    'framework',
    'freedom',
    'fundamental',
    'gai',
    'genai',
    'generative',
    'generators',
    'goal',
    'governance',
    'gpt',
    'grade',
    'graduate',
    'growth',
    'gsb',
    'guidance',
    'guidancemit',
    'guide',
    'guideline',
    'health',
    'hgse',
    'hipaa',
    'home',
    'honor',
    'hub',
    'human',
    'hybrid',
    'idea',
    'identifier',
    'impact',
    'implement',
    'implication',
    'improve',
    'inaccurate',
    'inclusive',
    'incorporate',
    'incorrect',
    'industry',
    'initiative',
    'innovation',
    'input',
    'inputting',
    'insight',
    'institutional',
    'instructional',
    'instructor',
    'instructors',
    'insurance',
    'integrate',
    'integrity',
    'intellectual',
    'job',
    'journey',
    'keyboard',
    'kit',
    'knowledge',
    'language',
    'languagemodel',
    'law',
    'learn',
    'legal',
    'level',
    'leverage',
    'library',
    'life',
    'logon',
    'market',
    'mba',
    'menu',
    'messaging',
    'mind',
    'misconduct',
    'model',
    'module',
    'news',
    'novice',
    'office',
    'open',
    'openai',
    'opportunity',
    'outcome',
    'output',
    'party',
    'patient',
    'pattern',
    'perform',
    'permitted',
    'personal',
    'plagiarism',
    'platform',
    'policy',
    'practice',
    'president',
    'principle',
    'privacy',
    'problem',
    'procured',
    'procurement',
    'product',
    'professor',
    'program',
    'prohibit',
    'project',
    'prompt',
    'property',
    'proposal',
    'protect',
    'protection',
    'provost',
    'public',
    'quizzes',
    'rapidly',
    'rationale',
    'recommendation',
    'record',
    'refine',
    'reflect',
    'registrar',
    'relevant',
    'remote',
    'research',
    'researcher',
    'residential',
    'resource',
    'respect',
    'retention',
    'review',
    'risk',
    'safe',
    'sanction',
    'scholar',
    'school',
    'science',
    'secure',
    'security',
    'senate',
    'sexual',
    'skill',
    'social',
    'software',
    'solution',
    'solve',
    'space',
    'stake',
    'statement',
    'strategy',
    'student',
    'study',
    'syllabus',
    'symposium',
    'system',
    'tailor',
    'teach',
    'teaching',
    'team',
    'technology',
    'template',
    'tool',
    'toolkit',
    'topic',
    'training',
    'transparency',
    'uclas',
    'ucs',
    'uit',
    'undergraduate',
    'university',
    'usage',
    'vice',
    'workshop',
    'world',
    'write',
    'writers'
]

### load data #################################################################

lemmatized_data_fname = f'out{os.sep}_raw_lower_rgx_entity_stemmed_stopped_long_freq0.txt'
lemmatized_data = pd.read_csv(lemmatized_data_fname, header=None, skip_blank_lines=False)
logger.info(f'Loaded: {lemmatized_data_fname}.')
logger.info(lemmatized_data.head())

chunks_fname = f'dat{os.sep}chunk{os.sep}nongwu_policy_combined.csv'
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

chunk_data_fname = f'dat{os.sep}nongwu_policy_keyword.csv'
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