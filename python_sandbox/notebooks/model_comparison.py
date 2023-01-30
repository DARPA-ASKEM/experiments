# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Model Comparison
#
# Compare models via MIRA concept matching

# %%
import requests
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
import lxml.etree as etree
import networkx as nx
import hypernetx as hnx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any

# %%
REST_URL_XDD = 'https://xdd.wisc.edu/api'
REST_URL_BIOMODELS = 'https://www.ebi.ac.uk/biomodels'
REST_URL_MIRA = 'http://34.230.33.149:8771/api'

# %%
# Get the models in MMT format
# Weitz2020, Okuonghae2020, Zongo2020, Giordano2020
# models_id = ['BIOMD0000000955', 'BIOMD0000000991', 'BIOMD0000000983', 'BIOMD0000000963']
# __, models, __ = next(os.walk('../../thin-thread-examples/biomodels'))
df_models = pd.read_csv('../../thin-thread-examples/models.csv')
models_id = df_models['further_info'].values

models_id = models_id[:24]

models = []
for info in models_id:
    path = f'../../thin-thread-examples/biomodels/{info}'
    
    if os.path.exists(path + '/model_mmt.json'):
        with open(path + '/model_mmt.json', 'r') as f:
            models.append(json.load(f))
    else:
        print(f'Error: {info}')

# %%
# Load manual example from Ben Gyori @ HMS/MIRA
# with open('../../thin-thread-examples/model_comparison/mira_comparison_scenario4.json', 'r') as f:
#     model_comparison = json.load(f)

# %%
%%time

# Get model comparison via MIRA API
# http://34.230.33.149:8771/docs#/modeling/model_comparison_api_model_comparison_post

payload = {'template_models': models}
res = requests.post(f'{REST_URL_MIRA}/model_comparison', json = payload)
if res.status_code == 200:
    model_comparison = res.json()
else:
    print(f'Error: {res.status_code}')

# 3 models: 11.4 s
# 4 models: 42.2 s
# 5 models: 75 s
# 6 models: 120 s
# 8 models: 129 s
# 9 models: 187 s
# 10 models: 259 s
# 12 models: 629 s
# 24 models: 1921 s

with open('../../thin-thread-examples/model_comparison/mira_comparison_thin_thread.json', 'w') as f:
    f.write(json.dumps(model_comparison, indent = 4))

# %%
# Plot result

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))

A = np.full((len(models_id), len(models_id)), fill_value = None, dtype = np.float64)
for x in model_comparison['similarity_scores']:
    i, j = x['models']
    A[j, i] = x['score']
    A[i, i] = 1.0

# Similarity scores
h = axes.matshow(A, vmin = 0, vmax = 1, cmap = 'cividis')

# Inter-graph edges

__ = axes.set_xticks(np.arange(len(models_id)), labels = [i.replace('BIOMD0000000', 'BIOMD') for i in models_id])
__ = axes.set_yticks(np.arange(len(models_id)), labels = [i.replace('BIOMD0000000', 'BIOMD') for i in models_id])

__ = axes.tick_params(axis = 'x', labelrotation = 90)

__ = plt.colorbar(h, ax = axes, use_gridspec = True, location = 'bottom')

fig.savefig('../figures/model_comparison_thin_thread.png', dpi = 150)


# %%
# 