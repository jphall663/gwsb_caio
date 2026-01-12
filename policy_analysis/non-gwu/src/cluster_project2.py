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
# /home/patrickh/Workspace/gwsb_caio/policy_analysis/non-gwu/src/cluster_project.py

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

all_fname = f'dat{os.sep}gwu_nongwu_policy_keyword_embed_simple_label.csv'
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
umap_fname = f'dat{os.sep}gwu_nongwu_policy_keyword_embed_umap.csv'
all_.to_csv(umap_fname, index=False)
logger.info(f'Saved: {umap_fname}.')

### plot ######################################################################

logger.info('----------- -----------')
logger.info('Plotting ...')

# nice names
legend_dict = {

    'GRC': 'MISC: Governance, Risk & Compliance',
    'GWU_DP': 'GWU: Data Protection',
    'TG': 'MISC: AI Tools & Guidelines',
    'GWU_OCR': 'GWU: Office of Ethics, Compliance & Risk', 
    'GWU_LAI': 'GWU Libraries: Teaching AI',
    'GWU_Provost': 'GWU: Provost Guidance',
    'NONGWU_HC': 'NON-GWU-MISC: Honor Code and Conduct',
    'NONGWU_T': 'NON-GWU-MISC: Teaching AI',
    'MIT_T': 'MIT: Teaching Guidance'

}
all_['Type'] = all_['Type'].map(legend_dict)

# init plotting
clusters = all_['Type'].unique()
colors = cm.get_cmap("tab10", len(clusters))  # Set2 palette
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
ax.set_title('Visual Map of AI Policies', fontsize=25)

# save
plt.tight_layout()
plot_fname = f'out{os.sep}res{os.sep}doc_clus_legend.png'
plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
logger.info(f'Saved: {plot_fname}.')


### cluster profiling #########################################################

logger.info('----------- -----------')
logger.info('Profiling and saving results ...')

### understand what is in each large cluster, but not in gwu

#oercs_list = []
grc_list = []
gwu_dp_list = []
tg_list = []
gwu_ocr_list = []
gwu_lai_list = []
gwu_provost_list = []
nongwu_hc_list = []
nongwu_t_list = []
mit_t_list = []

for key in profile_dict.keys():
    #if key == 'Berkeley: AI Risk Subcommittee':
    #    oercs_list += profile_dict[key]['cl_non_unique_keyword_list']
    if key == 'MISC: Governance, Risk & Compliance':
        grc_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'GWU: Data Protection':
        gwu_dp_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'MISC: AI Tools & Guidelines':
        tg_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'GWU: Office of Ethics, Compliance & Risk':
        gwu_ocr_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'GWU Libraries: Teaching AI':
        gwu_lai_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'GWU: Provost Guidance':
        gwu_provost_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'NON-GWU-MISC: Honor Code and Conduct':
        nongwu_hc_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'NON-GWU-MISC: Teaching AI': 
        nongwu_t_list += profile_dict[key]['cl_non_unique_keyword_list']
    elif key == 'MIT: Teaching Guidance':
        mit_t_list += profile_dict[key]['cl_non_unique_keyword_list']
        
#oercs_set = set(oercs_list)
grc_set = set(grc_list)
gwu_dp_set = set(gwu_dp_list)
tg_set = set(tg_list)
gwu_ocr_set = set(gwu_ocr_list)
gwu_lai_set = set(gwu_lai_list)
gwu_provost_set = set(gwu_provost_list)
nongwu_hc_set = set(nongwu_hc_list)
nongwu_t_set = set(nongwu_t_list)
mit_t_set = set(mit_t_list)

### create count list, convert each to csv, and save
### create word clouds

prefix_list = ['grc', 'gwu_dp', 'tg', 'gwu_ocr', 'gwu_lai', 'gwu_provost', 'nongwu_hc', 'nongwu_t', 'mit_t']
list_list = [grc_list, gwu_dp_list, tg_list, gwu_ocr_list, gwu_lai_list, gwu_provost_list, nongwu_hc_list, nongwu_t_list, mit_t_list]
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