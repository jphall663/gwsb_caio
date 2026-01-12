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

all_fname = f'dat{os.sep}gwu_nongwu_policy_keyword_embed.csv'
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


#'GWU_AIGuidanceandBestPractices_GWInformationTechnology_TheGeorgeWashingtonUniversity': 'GWU_AIGuidanceandBestPractices_GWInformationTechnology_TheGeorgeWashingtonUniversity',
#'GWU_AcceptableUseofITResourcesPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity': 'GWU_AcceptableUseofITResourcesPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity',
#'GWU_ArtificialIntelligence(AI)Evaluation&Status_GWInformationTechnology_TheGeorgeWashingtonUniversity': 'GWU_ArtificialIntelligence(AI)Evaluation&Status_GWInformationTechnology_TheGeorgeWashingtonUniversity',
#'GWU_CommunicatingYourGenAIExpectationstoYourStudents_Libraries&AcademicInnovation': 'GWU_CommunicatingYourGenAIExpectationstoYourStudents_Libraries&AcademicInnovation',
#'GWU_CybersecurityRiskPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity': 'GWU_CybersecurityRiskPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity',
#'GWU_DataClassificationGuide_GWInformationTechnology_TheGeorgeWashingtonUniversity': 'GWU_DataClassificationGuide_GWInformationTechnology_TheGeorgeWashingtonUniversity',
#'GWU_DataProtectionGuide_GWInformationTechnology_TheGeorgeWashingtonUniversity': 'GWU_DataProtectionGuide_GWInformationTechnology_TheGeorgeWashingtonUniversity',
#'GWU_DecidingonAppropriateUseofGenAIinAcademicClasses_Libraries&AcademicInnovation': 'GWU_DecidingonAppropriateUseofGenAIinAcademicClasses_Libraries&AcademicInnovation',
#'GWU_ExploreTools&Services_GWInformationTechnology_TheGeorgeWashingtonUniversity': 'GWU_ExploreTools&Services_GWInformationTechnology_TheGeorgeWashingtonUniversity',
#'GWU_GenerativeArtificialIntelligence(GenAI)_Libraries&AcademicInnovation': 'GWU_GenerativeArtificialIntelligence(GenAI)_Libraries&AcademicInnovation',
#'GWU_IdentityandAccessManagementPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity': 'GWU_IdentityandAccessManagementPolicy_OfficeofEthics,Compliance,andRisk_TheGeorgeWashingtonUniversity',
#'GWU_PrivacyConsiderationswhenusingVirtualMeetingandCollaborationPlatforms_GWPrivacyOffice_TheGeorgeWashingtonUniversity': 'GWU_PrivacyConsiderationswhenusingVirtualMeetingandCollaborationPlatforms_GWPrivacyOffice_TheGeorgeWashingtonUniversity',
#'GWU_PrivacyGuidanceforuseofArtificialIntelligence_GWPrivacyOffice_TheGeorgeWashingtonUniversity': 'GWU_PrivacyGuidanceforuseofArtificialIntelligence_GWPrivacyOffice_TheGeorgeWashingtonUniversity',
#'GWU_TeachingwithGenerativeAI_Libraries&AcademicInnovation': 'GWU_TeachingwithGenerativeAI_Libraries&AcademicInnovation',
#'GWU_additional_guidance_for_generative_ai_-_august_2023': 'GWU_additional_guidance_for_generative_ai_-_august_2023',
#'GWU_generative-artificial-intelligence-guidelines-april-2023': 'GWU_generative-artificial-intelligence-guidelines-april-2023',

#'NONGWU_SU25-GAI_PoliciesPractices': 'NONGWU_SU25-GAI_PoliciesPractices',
#'NONGWU_ai.nd.edu_policies-and-guidelines': 'NONGWU_ai.nd.edu_policies-and-guidelines',
#'NONGWU_ai.universityofcalifornia.edu_governance-transparency': 'NONGWU_ai.universityofcalifornia.edu_governance-transparency',
#'NONGWU_ai.universityofcalifornia.edu_governance-transparency_applicable-law-and-policy.': 'NONGWU_ai.universityofcalifornia.edu_governance-transparency_applicable-law-and-policy.',
#'NONGWU_ai.yale.edu': 'NONGWU_ai.yale.edu',
'NONGWU_cndls.georgetown.edu_resources_ai': 'NONGWU_cndls.georgetown.edu_resources_ai',
'NONGWU_cndls.georgetown.edu_resources_syllabus-policies_ai-and-homework-support': 'NONGWU_cndls.georgetown.edu_resources_syllabus-policies_ai-and-homework-support',
#'NONGWU_communitystandards.stanford.edu_generative-ai-policy-guidance': 'NONGWU_communitystandards.stanford.edu_generative-ai-policy-guidance',
#'NONGWU_ctl.columbia.edu_resources-and-technology_resources_ai-tools': 'NONGWU_ctl.columbia.edu_resources-and-technology_resources_ai-tools',
#'NONGWU_dtei.uci.edu_generative-ai': 'NONGWU_dtei.uci.edu_generative-ai',
#'NONGWU_genai.ucla.edu': 'NONGWU_genai.ucla.edu',
#'NONGWU_genai.ucla.edu_guiding-principles-responsible-use': 'NONGWU_genai.ucla.edu_guiding-principles-responsible-use',
#'NONGWU_guides.library.georgetown.edu_ai': 'NONGWU_guides.library.georgetown.edu_ai',
#'NONGWU_honorcode.nd.edu_ai-recommendations-for-instructors': 'NONGWU_honorcode.nd.edu_ai-recommendations-for-instructors',
#'NONGWU_honorcode.nd.edu_generative-ai-policy-for-students-august-2023': 'NONGWU_honorcode.nd.edu_generative-ai-policy-for-students-august-2023',
#'NONGWU_ist.mit.edu_ai-guidance': 'NONGWU_ist.mit.edu_ai-guidance',
#'NONGWU_libguides.usc.edu_generative-AI_home': 'NONGWU_libguides.usc.edu_generative-AI_home',
#'NONGWU_oercs.berkeley.edu_CERC-AIR': 'NONGWU_oercs.berkeley.edu_CERC-AIR',
#'NONGWU_oercs.berkeley.edu_appropriate-use-generative-ai-tools': 'NONGWU_oercs.berkeley.edu_appropriate-use-generative-ai-tools',
#'NONGWU_poorvucenter.yale.edu_AIguidance': 'NONGWU_poorvucenter.yale.edu_AIguidance',
#'NONGWU_provost.harvard.edu_guidelines-using-chatgpt-and-other-generative-ai-tools-harva': 'NONGWU_provost.harvard.edu_guidelines-using-chatgpt-and-other-generative-ai-tools-harva',
#'NONGWU_provost.yale.edu_news_guidelines-use-generative-ai-tools': 'NONGWU_provost.yale.edu_news_guidelines-use-generative-ai-tools',3
#'NONGWU_registrar.gse.harvard.edu_AI-policy': 'NONGWU_registrar.gse.harvard.edu_AI-policy',
'NONGWU_senate.ucla.edu_news_teaching-guidance-chatgpt-and-related-ai-developments': 'NONGWU_senate.ucla.edu_news_teaching-guidance-chatgpt-and-related-ai-developments',
#'NONGWU_teaching.ucla.edu_resources_teaching-guides_using-generative-ai-reflectively-and': 'NONGWU_teaching.ucla.edu_resources_teaching-guides_using-generative-ai-reflectively-and',
#'NONGWU_teaching.washington.edu_course-design_ai': 'NONGWU_teaching.washington.edu_course-design_ai',
#'NONGWU_teaching.washington.edu_course-design_ai_sample-ai-syllabus-statements': 'NONGWU_teaching.washington.edu_course-design_ai_sample-ai-syllabus-statements',
#'NONGWU_teachingcommons.stanford.edu_teaching-guides_artificial-intelligence-teaching-gu': 'NONGWU_teachingcommons.stanford.edu_teaching-guides_artificial-intelligence-teaching-gu',
#'NONGWU_technology.berkeley.edu_AI': 'NONGWU_technology.berkeley.edu_AI',
#'NONGWU_tlhub.stanford.edu_docs_course-policies-on-generative-ai-use': 'NONGWU_tlhub.stanford.edu_docs_course-policies-on-generative-ai-use',
#'NONGWU_tll.mit.edu_teaching-resources_course-design_gen-ai-your-course': 'NONGWU_tll.mit.edu_teaching-resources_course-design_gen-ai-your-course',
#'NONGWU_uit.stanford.edu_security_responsibleai': 'NONGWU_uit.stanford.edu_security_responsibleai',
#'NONGWU_www.hbs.edu_mba_handbook_standards-of-conduct_academic_Pages_chatgpt-and-ai.aspx': 'NONGWU_www.hbs.edu_mba_handbook_standards-of-conduct_academic_Pages_chatgpt-and-ai.aspx',
#'NONGWU_yaledata.yale.edu_yale-university-ai-guidelines-staff': 'NONGWU_yaledata.yale.edu_yale-university-ai-guidelines-staff'
}
all_['Type'] = all_['Type'].map(legend_dict)

# init plotting
clusters = all_['Type'].unique()
colors = cm.get_cmap("Set2", len(clusters))  # Set2 palette
centroids = {}
fig, ax = plt.subplots(figsize=(40, 32))
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
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.65)
    )


# remove ticks, values, and surrounding box
ax.set_xticks([]); ax.set_yticks([])
ax.set_xticklabels([]); ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
#ax.legend(title='', loc='best', fontsize=20)

# title
ax.set_title('Visual Map of AI Policies', fontsize=25)

# save
plt.tight_layout()
plot_fname = f'out{os.sep}res{os.sep}doc_clus.png'
plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
logger.info(f'Saved: {plot_fname}.')


### cluster profiling #########################################################

"""

logger.info('----------- -----------')
logger.info('Profiling and saving results ...')

### understand what is in each large cluster, but not in gwu

gwu_list = []
for key in profile_dict.keys():
    if key.startswith('GWU'):
        gwu_list += profile_dict[key]['cl_non_unique_keyword_list']

gwu_set = set(gwu_list)        

non_gwu_list = []
for key in profile_dict.keys():
    if key.startswith('NONGWU'):
        non_gwu_list += profile_dict[key]['cl_non_unique_keyword_list']

non_gwu_set = set(non_gwu_list)   

### create count list, convert each to csv, and save
### create word clouds

prefix_list = ['gwu', 'non_gwu', 'eval_approved', 'guidance', 'data', 'oecr', 'meetings']
list_list = [gwu_list, non_gwu_list]

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

"""
    
# end timer
toc = time.time() - tic
logger.info(f'All tasks performed in {toc:.2f} s.')

