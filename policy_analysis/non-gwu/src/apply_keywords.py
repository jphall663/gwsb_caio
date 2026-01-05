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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/gwu/src/apply_keywords.py

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
    'academic',
    'acceptable',
    'accessibility',
    'activity',
    'adapt',
    'administrative',
    'adobe',
    'aggregate',
    'applicable',
    'approval',
    'artificialintelligence',
    'assignment',
    'assistance',
    'attendees',
    'authorize',
    'barrier',
    'capability',
    'chat',
    'chatbots',
    'chatgpt',
    'class',
    'classification',
    'classroom',
    'cloud',
    'collaboration',
    'communicate',
    'companion',
    'compliance',
    'computer',
    'consultation',
    'context',
    'control',
    'create',
    'custody',
    'cybersecurity',
    'data',
    'device',
    'digital',
    'draft',
    'electronic',
    'employee',
    'encryption',
    'ethic',
    'evaluate',
    'evaluation',
    'event',
    'excellence',
    'expectation',
    'explore',
    'facility',
    'faculty',
    'final',
    'gai',
    'gelmangwu',
    'genai',
    'generative',
    'guidance',
    'guide',
    'guideline',
    'gwid',
    'gws',
    'host',
    'human',
    'ias',
    'idea',
    'identity',
    'install',
    'instructors',
    'integrity',
    'intellectual',
    'invite',
    'it',
    'ithelpgwu',
    'knowledge',
    'language',
    'languagemodels',
    'law',
    'learn',
    'legitimate',
    'level',
    'library',
    'loss',
    'measure',
    'mobile',
    'model',
    'objective',
    'office',
    'output',
    'owned',
    'permitted',
    'personal',
    'phone',
    'physical',
    'pii',
    'platform',
    'policy',
    'practice',
    'prints',
    'privacy',
    'product',
    'program',
    'prompt',
    'protect',
    'protection',
    'provost',
    'public',
    'quality',
    'quiz',
    'record',
    'regulate',
    'requirement',
    'research',
    'researcher',
    'resource',
    'restrict',
    'review',
    'risk',
    'room',
    'scan',
    'school',
    'secure',
    'security',
    'sensitivity',
    'session',
    'skill',
    'software',
    'step',
    'strongly',
    'student',
    'style',
    'success',
    'system',
    'teach',
    'technology',
    'telehealth',
    'tls',
    'tool',
    'transmit',
    'unacceptable',
    'unauthorized',
    'university',
    'usiness',
    'verify',
    'violation',
    'virtual',
    'workshop',
    'write',
    'zoom'

]

### load data #################################################################

lemmatized_data_fname = f'out{os.sep}_raw_lower_rgx_entity_stemmed_stopped_long_freq0.txt'
lemmatized_data = pd.read_csv(lemmatized_data_fname, header=None, skip_blank_lines=False)
logger.info(f'Loaded: {lemmatized_data_fname}.')
logger.info(lemmatized_data.head())

chunks_fname = f'dat{os.sep}chunk{os.sep}existing_policy_combined.csv'
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

chunk_data_fname = f'dat{os.sep}existing_policy_keyword.csv'
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
wc_fname = f'out{os.sep}res{os.sep}existing_policy_key_word_cloud_hi_4k.png'
wordcloud.to_file(wc_fname)
logger.info(f'Saved: {wc_fname}.')