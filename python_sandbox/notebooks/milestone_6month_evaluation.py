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
# Population distribution for Scenario 1.2

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
# Scenario 3:
# Mapping between TA1 dataset to model state variables

ta1 = {}
ta1['cases-deaths'] = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-cases-deaths.csv')
ta1['cases-hosp'] = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-cases-hospitalized-by-age.csv')
ta1['hosp'] = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-hospitalizations.csv')
ta1['vacc'] = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-vaccinations.csv')
ta1['pop'] = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-2021-population-age-stratified.csv')

# %%
# N, S, I, R, D, H, V, I_age, N_age

t1 = ta1['cases-deaths']['date'][14:].values
I = ta1['cases-deaths']['new_confirmed'][14:].values + ta1['cases-deaths']['cumulative_confirmed'][14:].values - ta1['cases-deaths']['cumulative_confirmed'][:-14].values
R = ta1['cases-deaths']['cumulative_confirmed'][:-14].values
D = ta1['cases-deaths']['cumulative_deceased'].values

t2 = ta1['vacc']['date'].values
V = ta1['vacc']['cumulative_persons_vaccinated'].values
H = ta1['hosp']['current_hospitalized_patients'].values

# Age breakdown
# 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79
t3 = ta1['cases-hosp']['date'][14:].values
j = [f'cumulative_confirmed_age_{i}' for i in range(0, 8)]
k = [f'new_confirmed_age_{i}' for i in range(0, 8)]
I_age = ta1['cases-hosp'][k].values[14:, :] + ta1['cases-hosp'][j].values[14:, :] - ta1['cases-hosp'][j].values[:-14, :]

# Overlap time range
t_start = '2020-12-13'
t_end = '2022-09-02'

i = np.where(t1 == t_start)[0].item()
j = np.where(t1 == t_end)[0].item() + 1
t1 = t1[i:j]
I = I[i:j]
R = R[i:j]
D = D[i:j]

N = ta1['pop']['Population'].loc[0]
N_age = ta1['pop']['Population'].values[1:21][0::2] + ta1['pop']['Population'].values[1:21][1::2]
N_age = np.append(N_age, ta1['pop']['Population'][21])
N_age = np.tile(N_age, (j - i, 1))


i = np.where(t2 == t_start)[0].item()
j = np.where(t2 == t_end)[0].item() + 1
t2 = t2[i:j]
V = V[i:j]
H = H[i:j]

i = np.where(t3 == t_start)[0].item()
j = np.where(t3 == t_end)[0].item() + 1
t3 = t3[i:j]
I_age = I_age[i:j, :]


data = np.concatenate((t1[:, np.newaxis], I[:, np.newaxis], R[:, np.newaxis], D[:, np.newaxis], V[:, np.newaxis], H[:, np.newaxis], I_age, N_age), axis = 1)

df = pd.DataFrame(
    data, 
    columns = [
        'date', 
        'I', 
        'R', 
        'D', 
        'V', 
        'H', 
        'I_0-9', 'I_10-19', 'I_20-29', 'I_30-39', 
        'I_40-49', 'I_50-59', 'I_60-69', 'I_70-79', 
        'N_0-9', 'N_10-19', 'N_20-29', 'N_30-39', 
        'N_40-49', 'N_50-59', 'N_60-69', 'N_70-79',
        'N_80-89', 'N_90-99', 'N_100+'
    ]
)

df.to_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-IRDVHN_age.csv')

# %%


# %%
