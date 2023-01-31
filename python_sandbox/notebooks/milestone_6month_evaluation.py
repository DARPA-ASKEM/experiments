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
# Scenario 2.a.
#
# Model comparison between SIDARTHE and SIDARTHE-V models

path = f'../../thin-thread-examples/milestone_6month/evaluation/indra'
models = {}
for s in ['/scenario_2/sidarthe', '/scenario_2/sidarthe-v']:
    with open(path + s + '/model_mmt.json', 'r') as f:
        models[s.split('/')[2]] = json.load(f)


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

