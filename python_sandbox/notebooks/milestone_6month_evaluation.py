# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# ASKEM Hackathon (Jan 2023)
#
# Plan:
# 1. Scenario 1.1 model (BIOMD...991)
# 2. Add hospitalization
# 3. Scenario 2.1 parameter values and initial conditions

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
# Scenario 2.1.a.
#
# Model comparison between SIDARTHE and SIDARTHE-V models

path = f'../../thin-thread-examples/milestone_6month/evaluation/indra'
models = {}
for s in ['/scenario_2/sidarthe', '/scenario_2/sidarthe-v']:
    with open(path + s + '/model_mmt.json', 'r') as f:
        models[s.split('/')[2]] = json.load(f)

# %%
# Get model comparison via MIRA API
# http://34.230.33.149:8771/docs#/modeling/model_comparison_api_model_comparison_post
model_comparison = {} 
payload = {'template_models': list(models.values())}
res = requests.post(f'{REST_URL_MIRA}/model_comparison', json = payload)
if res.status_code == 200:
    model_comparison[('sidarthe', 'sidarthe-v')] = res.json()
else:
    print(f'Error: {res.status_code}')


with open(path + '/scenario_2/model_comparison.json', 'w') as f:
    f.write(json.dumps(model_comparison[('sidarthe', 'sidarthe-v')], indent = 4))

# %%
# Scenario 3.2
#
# Model comparison between BIOMD958 and BIOMD960 models

for s in ['/scenario_3/biomd958', '/scenario_3/biomd960']:
    with open(path + s + '/model_mmt.json', 'r') as f:
        models[s.split('/')[2]] = json.load(f)

payload = {'template_models': [v for k, v in models.items() if k in ('biomd958', 'biomd960')]}
res = requests.post(f'{REST_URL_MIRA}/model_comparison', json = payload)
if res.status_code == 200:
    model_comparison[('biomd958', 'biomd960')] = res.json()
else:
    print(f'Error: {res.status_code}')


with open(path + '/scenario_3/model_comparison.json', 'w') as f:
    f.write(json.dumps(model_comparison[('biomd958', 'biomd960')], indent = 4))


# %%

belgium = pd.read_csv('../../python_sandbox/data/hackathon_20230126/2016_belgium_population_by_age.csv')
india = pd.read_csv('../../python_sandbox/data/hackathon_20230126/2016_india_population_by_age.csv')

fig, axes = plt.subplots(1, 2, figsize = (8, 4))
fig.subplots_adjust(wspace = 0.1)
ylab = list(belgium.columns[2:].to_numpy())

y = range(len(ylab))
__ = axes[0].barh(y, belgium.iloc[0, 2:] / belgium.iloc[0, 1], orientation = 'horizontal')
__ = axes[1].barh(y, india.iloc[0, 2:] / india.iloc[0, 1], orientation = 'horizontal')

for ax, c in zip(axes, ('Belgium', 'India')):
    __ = plt.setp(ax, title = c, xlabel = 'Pop Fraction', ylabel = 'Age Groups', xlim = [0, 0.10])
    __ = ax.set_yticks(y, labels = ylab)

__ = plt.setp(axes[1], ylabel = '')
axes[1].tick_params('y', labelleft = False)

# %%
