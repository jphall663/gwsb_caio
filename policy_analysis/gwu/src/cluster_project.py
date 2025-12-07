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

### imports and configs #######################################################

from collections import Counter

from logging_utils import get_logger
logger = get_logger(__name__)

import matplotlib.cm as cm
from matplotlib import MatplotlibDeprecationWarning
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import normalize
import time
import umap

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=MatplotlibDeprecationWarning)

from wordcloud import WordCloud

embedding_p = 1536

tic = time.time()

### load data #################################################################

logger.info('----------- -----------')
logger.info(f'Loading data ...')

embedding_names = ['dim_' + str(i) for i in range(0, embedding_p)]
valid_cols = ['Type', 'ID', 'Keywords'] + embedding_names

all_fname = f'dat{os.sep}existing_policy_keyword_embed.csv'
all_ = pd.read_csv(all_fname)
logger.info(f'Loaded: {all_fname}.')

all_['Keywords'] = all_['Keywords'].fillna('')
logger.info(f'Any missing values: {all_.isna().any().any()}.')
all_['Keywords'] = all_['Keywords'].str.split(',')

N = all_.shape[0]
logger.info(f'Correct data shape: {all_.shape == (792, 1539)}.')

### extract embeddings and normalize ##########################################

logger.info('----------- -----------')
logger.info(f'Extracting and normalizing embeddings ...')

# extract
X = all_[embedding_names].values

# handle 0's
eps = 0.000000001
X[X == 0.0] = eps

# normalize
X = normalize(X, norm="l2", axis=1)

# show results
logger.info('Embeddings head:')
logger.info(X[0:5, :])
logger.info(f'Embeddings correct shape: {str(X.shape == (N, embedding_p))}')

### perform umap ##############################################################

logger.info('----------- -----------')
logger.info('Performing UMAP ... ')

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=0
)

X_2d = reducer.fit_transform(X)  # shape (n, 2)

logger.info('UMAP results head:')
logger.info(X_2d[0:5, :])

# add umap results to data
all_['UMAP_D1'] = X_2d[:, 0]
all_['UMAP_D2'] = X_2d[:, 1]

# save umap results
umap_fname = f'dat{os.sep}existing_policy_keyword_embed_umap.csv'
all_.to_csv(umap_fname, index=False)
logger.info(f'Saved: {umap_fname}.')

### plot ######################################################################

logger.info('----------- -----------')
logger.info('Plotting ...')

# nice names
legend_dict = {
    'Acceptable Use of IT Resources Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'Acceptable Use (OECR)',
    'additional_guidance_for_generative_ai_-_august_2023': 'Provost 2',
    'AI Guidance and Best Practices _ GW Information Technology _ The George Washington University': 'AI Guidance and Best Pratice (IT)',
    'Artificial Intelligence (AI) Evaluation & Status _ GW Information Technology _ The George Washington University': 'Tool Evaluation (IT)',
    'Communicating Your GenAI Expectations to Your Students _ Libraries & Academic Innovation': 'Communicating Expecations (Libraries)',
    'Cybersecurity Risk Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'Cybersecurity Risk (OECR)',
    'Data Classification Guide _ GW Information Technology _ The George Washington University': 'Data Classification (IT)',
    'Data Protection Guide _ GW Information Technology _ The George Washington University': 'Data Protection (IT)',
    'Deciding on Appropriate Use of GenAI in Academic Classes _ Libraries & Academic Innovation': 'Deciding on Use (Libraries)',
    'Explore Tools & Services _ GW Information Technology _ The George Washington University': 'Approved Tools (IT)',
    'Generative Artificial Intelligence (GenAI) _ Libraries & Academic Innovation': 'Generative AI (Libraries)',
    'generative-artificial-intelligence-guidelines-april-2023': 'Provost 1', 
    'Identity and Access Management Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'IAM (OECR)',
    'Privacy Considerations when using Virtual Meeting and Collaboration Platforms _ GW Privacy Office _ The George Washington University': 'Privacy Guidance: Meetings',
    'Privacy Guidance for use of Artificial Intelligence _ GW Privacy Office _ The George Washington University': 'Privacy Guidance',
    'Teaching with Generative AI _ Libraries & Academic Innovation': 'Teaching with Gen AI (Libraries)'
}
all_['Type'] = all_['Type'].map(legend_dict)

# init plotting
clusters = all_['Type'].unique()
colors = cm.get_cmap("Set2", len(clusters))  # Set2 palette
centroids = {}
fig, ax = plt.subplots(figsize=(20, 16))
profile_dict = {}

# plot all clusters
for i, cl in enumerate(clusters):

    mask = all_['Type'] == cl

    plt.scatter(
        all_.loc[mask, 'UMAP_D1'],
        all_.loc[mask, 'UMAP_D2'],
        s=50,
        alpha=0.6,
        color=colors(i),
        label=cl
    )

    # label coordinates
    cx = all_.loc[mask, 'UMAP_D1'].median()
    cy = all_.loc[mask, 'UMAP_D2'].median()
    centroids[cl] = (cx, cy)

    # capture profiling information
    profile_dict[cl] = {'cl_non_unique_keyword_list': None,
                        'cl_non_unique_keyword_set': None}
    profile_dict[cl]['cl_non_unique_keyword_list'] = sum(all_.loc[mask, 'Keywords'], [])
    profile_dict[cl]['cl_non_unique_keyword_list'] = [w.strip() for w in profile_dict[cl]['cl_non_unique_keyword_list']]
    profile_dict[cl]['cl_non_unique_keyword_list'] = sorted([w for w in profile_dict[cl]['cl_non_unique_keyword_list'] if w != ''])
    logger.info(f"{cl} list: {profile_dict[cl]['cl_non_unique_keyword_list']}")
    profile_dict[cl]['cl_non_unique_keyword_set'] = set(profile_dict[cl]['cl_non_unique_keyword_list'])
    logger.info(f"{cl} set: {profile_dict[cl]['cl_non_unique_keyword_set']}")

# annotate centroids with a small text box
for _, cl in enumerate(clusters):
    cx, cy = centroids[cl]
    plt.text(
        cx, cy, str(cl),
        ha='center', va='center',
        fontsize=14,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.85)
    )


# remove ticks, values, and surrounding box
ax.set_xticks([]); ax.set_yticks([])
ax.set_xticklabels([]); ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend(title='', loc='best', fontsize=20)

# title
ax.set_title('Visual Map of GWU AI Policies', fontsize=25)

# save
plt.tight_layout()
plot_fname = f'out{os.sep}res{os.sep}doc_clus_legend.png'
plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
logger.info(f'Saved: {plot_fname}.')


### cluster profiling #########################################################

logger.info('----------- -----------')
logger.info('Profiling and saving results ...')

### understand what is in each large cluster, but not in sn

legend_dict = {
    'Acceptable Use of IT Resources Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'Acceptable Use (OECR)',
    'additional_guidance_for_generative_ai_-_august_2023': 'Provost 2',
    'AI Guidance and Best Practices _ GW Information Technology _ The George Washington University': 'AI Guidance and Best Pratice (IT)',
    'Artificial Intelligence (AI) Evaluation & Status _ GW Information Technology _ The George Washington University': 'Tool Evaluation (IT)',
    'Communicating Your GenAI Expectations to Your Students _ Libraries & Academic Innovation': 'Communicating Expecations (Libraries)',
    'Cybersecurity Risk Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'Cybersecurity Risk (OECR)',
    'Data Classification Guide _ GW Information Technology _ The George Washington University': 'Data Classification (IT)',
    'Data Protection Guide _ GW Information Technology _ The George Washington University': 'Data Protection (IT)',
    'Deciding on Appropriate Use of GenAI in Academic Classes _ Libraries & Academic Innovation': 'Deciding on Use (Libraries)',
    'Explore Tools & Services _ GW Information Technology _ The George Washington University': 'Approved Tools (IT)',
    'Generative Artificial Intelligence (GenAI) _ Libraries & Academic Innovation': 'Generative AI (Libraries)',
    'generative-artificial-intelligence-guidelines-april-2023': 'Provost 1', 
    'Identity and Access Management Policy _ Office of Ethics, Compliance, and Risk _ The George Washington University': 'IAM (OECR)',
    'Privacy Considerations when using Virtual Meeting and Collaboration Platforms _ GW Privacy Office _ The George Washington University': 'Privacy Guidance: Meetings',
    'Privacy Guidance for use of Artificial Intelligence _ GW Privacy Office _ The George Washington University': 'Privacy Guidance',
    'Teaching with Generative AI _ Libraries & Academic Innovation': 'Teaching with Gen AI (Libraries)'
}




provost_list = profile_dict['Provost 1']['cl_non_unique_keyword_list'] +\
               profile_dict['Provost 2']['cl_non_unique_keyword_list'] 
provost_set = set(provost_list)

libraries_list = profile_dict['Communicating Expecations (Libraries)']['cl_non_unique_keyword_list'] + \
                 profile_dict['Deciding on Use (Libraries)']['cl_non_unique_keyword_list'] +\
                 profile_dict['Teaching with Gen AI (Libraries)']['cl_non_unique_keyword_list'] +\
                 profile_dict['Generative AI (Libraries)']['cl_non_unique_keyword_list']
libraries_set = set(libraries_list)

eval_approved_list = profile_dict['Tool Evaluation (IT)']['cl_non_unique_keyword_list'] + \
                     profile_dict['Approved Tools (IT)']['cl_non_unique_keyword_list']
eval_approved_set = set(eval_approved_list)

guidance_list = profile_dict['AI Guidance and Best Pratice (IT)']['cl_non_unique_keyword_list'] + \
                profile_dict['Privacy Guidance']['cl_non_unique_keyword_list']
guidance_set = set(guidance_list)

data_list = profile_dict['Data Classification (IT)']['cl_non_unique_keyword_list'] + \
            profile_dict['Data Protection (IT)']['cl_non_unique_keyword_list']
data_set = set(data_list)

oecr_list = profile_dict['Acceptable Use (OECR)']['cl_non_unique_keyword_list'] + \
            profile_dict['Cybersecurity Risk (OECR)']['cl_non_unique_keyword_list'] +\
            profile_dict['IAM (OECR)']['cl_non_unique_keyword_list']
oecr_set = set(oecr_list)

meetings_list = profile_dict['Privacy Guidance: Meetings']['cl_non_unique_keyword_list']
meetings_set = set(meetings_list)

### create count list, convert each to csv, and save
### create word clouds

prefix_list = ['provost', 'libraries', 'eval_approved', 'guidance', 'data', 'oecr', 'meetings']
list_list = [provost_list, libraries_list, eval_approved_list, guidance_list, data_list, oecr_list, meetings_list]

for i, list_ in enumerate(list_list):

    logger.info(f'Generating {prefix_list[i]} count list ...')
    counts = Counter(list_)
    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['count']).reset_index()
    counts_df.rename(columns={'index': 'item'}, inplace=True)
    counts_df_fname = f'out{os.sep}res{os.sep}{prefix_list[i]}_unique_counts.csv'
    counts_df.to_csv(counts_df_fname, index=False)
    logger.info(f'Saved: {counts_df_fname}.')

    logger.info(f'Generating {prefix_list[i]} word cloud ...')

    wc_text = ' '.join(list_)

    wordcloud = WordCloud(
        width=4000,
        height=2000,
        max_words=200,       # maximum number of words shown
        min_font_size=7,     # minimum font size
        background_color='white',
        colormap='Set2',  # pastel colormap
        collocations=False
    ).generate(wc_text)

    # save to file
    wc_fname = f'out{os.sep}res{os.sep}{prefix_list[i]}_unique_key_word_cloud_hi_4k.png'
    wordcloud.to_file(wc_fname)
    logger.info(f'Saved: {wc_fname}.')


# end timer
toc = time.time() - tic
logger.info(f'All tasks performed in {toc:.2f} s.')
