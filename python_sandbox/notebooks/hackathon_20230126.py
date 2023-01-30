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
# Grab Scenario 1.1 model from MIRA

doi = '10.3389/fpubh.2020.00230'
source = 'biomodels'
info = 'BIOMD0000000974'
path = f'../../thin-thread-examples/hackathon20230126/{source}/{info}'

# Get xDD ID
res = requests.get(f'{REST_URL_XDD}/articles?doi={doi}')
if res.status_code == 200:
    xdd_gddid = res.json()['success']['data'][0]['_gddid']
else:
    xdd_gddid = None

res = requests.get(f'{REST_URL_BIOMODELS}/model/files/{info}?format=json')
if res.status_code == 200:

    # SBML file
    main_filename = res.json()['main'][0]['name']
    additional_filenames = [f['name'] for f in res.json()['additional']]
    res = requests.get(f'{REST_URL_BIOMODELS}/model/download/{info}?filename={main_filename}')
    if res.status_code == 200:
        model_sbml = res.content
    else:
        model_sbml = None

    # Additional files
    model_add = []
    for filename in additional_filenames:
        res = requests.get(f'{REST_URL_BIOMODELS}/model/download/{info}?filename={filename}')
        if res.status_code == 200:
            model_add.append(res.content)


res = requests.get(f'{REST_URL_MIRA}/{source}/{info}')
if res.status_code == 200:
    model_mmt = res.json()
    model_mmt_templates = {'templates': model_mmt['templates']}
    model_mmt_parameters = {'parameters': model_mmt['parameters']}

    # Initial conditions
    # Find all state variables
    state_vars = [t['subject'] for t in model_mmt['templates'] if 'subject' in t.keys()]
    state_vars.extend([t['outcome'] for t in model_mmt['templates'] if 'outcome' in t.keys()])
    state_vars.extend([i for t in model_mmt['templates'] if 'controllers' in t.keys() for i in t['controllers']])
    state_vars.extend([t['controller'] for t in model_mmt['templates'] if 'controller' in t.keys()])
    state_vars_uniq = {hash(json.dumps(v, sort_keys = True, default = str, ensure_ascii = True).encode()): v for v in state_vars}
    model_mmt_initials = {'initials': {var['name']: {**var, **{'value': None}} for var in state_vars_uniq.values()}}

    # Populate with given values
    for k, v in model_mmt['initials'].items():
        if k in model_mmt_initials['initials'].keys():
            model_mmt_initials['initials'][k]['value'] = v

    # Get Petri net
    res = requests.post(f'{REST_URL_MIRA}/to_petrinet', json = model_mmt)
    if res.status_code == 200:
        model_petri = res.json()
    else:
        model_petri = None


# Create artifact directory if not exist
if os.path.exists(path) == False:
    os.mkdir(path)


# Write artifact files
for data, filename in zip([doi, xdd_gddid, model_sbml, model_mmt, model_mmt_templates, model_mmt_parameters, model_mmt_initials, model_petri], ['document_doi.txt', 'document_xdd_gddid.txt', main_filename, 'model_mmt.json', 'model_mmt_templates.json', 'model_mmt_parameters.json', 'model_mmt_initials.json', 'model_petri.json']):

    if data != None:
        
        # SBML XML file
        if filename.split('.')[-1] == 'xml':

            # `src` directory
            if os.path.exists(path + '/src') == False:
                os.mkdir(path + '/src')
            
            # `src/main` directory
            if os.path.exists(path + '/src/main') == False:
                os.mkdir(path + '/src/main')
            
            with open(path + f'/src/main/{filename}', 'wb') as f:
                f.write(data)

        else:
            with open(path + f'/{filename}', 'w') as f:
                if isinstance(data, dict):
                    f.write(json.dumps(data, indent = 4))
                else:
                    f.write(data)

    else:
        print(f'Error: {info} {filename} data = None')

# Write any additional source files
for data, filename in zip(model_add, additional_filenames):

    # `src/additional` directory
    if os.path.exists(path + '/src/additional') == False:
        os.mkdir(path + '/src/additional')

    with open(path + f'/src/additional/{filename}', 'wb') as f:
        f.write(data)

# %%
# Note:
# * Fix `Total_population` issue
# * Missing `mu` and `Lambda` natural death and birth process
# * Add hospitalization processes (I -h-> H, H -r-> R)

# %%
# Load models under the different representations

# Build a source-target DataFrame from MIRA JSON objects
def build_mira_df(model: dict, rep: str) -> pd.DataFrame:

    if rep == 'MMT':

        x = {
            'source': [],
            'source_type': [], 
            'target': [],
            'target_type': [],
            'edge_type': []
        }


        i = 0
        for l in model:

            # subject -> process
            x['source'].append(l['subject']['name'])
            x['source_type'].append(l['subject'])
            x['target'].append(i)
            x['target_type'].append(l['type'])
            x['edge_type'].append('subject')
            

            # process -> outcome
            x['source'].append(i)
            x['source_type'].append(l['type'])
            x['target'].append(l['outcome']['name'])
            x['target_type'].append(l['outcome'])
            x['edge_type'].append('outcome')

            # control -> process
            if 'controllers' in l.keys():
                for c in l['controllers']:
                    x['source'].append(c['name'])
                    x['source_type'].append(c)
                    x['target'].append(i)
                    x['target_type'].append(l['type'])
                    x['edge_type'].append('control')

            i += 1

    elif rep == 'Petri':

        map_petrinet_names = {'S': {(i + 1): s['sname'] for i, s in enumerate(model['S'])}, 'T': {(j + 1): s['tname'] for j, s in enumerate(model['T'])}}

        x = {
            'source': [map_petrinet_names['S'][d['is']] for d in model['I']] + [map_petrinet_names['T'][d['ot']] for d in model['O']],
            'source_type': ['S' for d in model['I']] + ['T' for d in model['O']],
            'target': [map_petrinet_names['T'][d['it']] for d in model['I']] + [map_petrinet_names['S'][d['os']] for d in model['O']],
            'target_type': ['T' for d in model['I']] + ['S' for d in model['O']],
            'edge_type': ['I' for d in model['I']] + ['O' for d in model['O']]
        }

    else:

        x = {}

    return pd.DataFrame(x)


# Build NetworkX graph from source-target DataFrame
def build_graph(df: pd.DataFrame) -> nx.DiGraph:

    G = nx.DiGraph()
    for i in ['source', 'target']:
        G.add_nodes_from([(node, {'type': node_type}) for __, (node, node_type) in df[[i, f"{i}_type"]].iterrows()])

    G.add_edges_from([(edge['source'], edge['target'], {'type': edge['edge_type']}) for __, edge in df.iterrows()])
    return G

# Build HyperNetX hypergraph from MIRA JSON object
def build_hypergraph(model: dict, rep: str) -> hnx.Hypergraph:

    if rep == 'MMT':

        h = {
            i: [he['subject']['name'], he['outcome']['name']] if 'controllers' not in he.keys() else [he['subject']['name'], he['outcome']['name']] + [n['name'] for n in he['controllers']]
            for i, he in enumerate(model)
        }

    elif rep == 'Petri':

        map_i_tname = {i + 1: t['tname'] for i, t in enumerate(model['T'])}
        map_i_sname = {i + 1: s['sname'] for i, s in enumerate(model['S'])}
        h = {
            t['tname']: [map_i_sname[edge['is']] for edge in model['I'] if map_i_tname[edge['it']] == t['tname']] + [map_i_sname[edge['os']] for edge in model['O'] if map_i_tname[edge['ot']] == t['tname']] 
            for t in model['T']
        }
        
    else:
        h = {}

    H = hnx.Hypergraph(h)

    return H

def draw_graph(G: nx.DiGraph, ax: Optional[Any] = None, node_type: Optional[list] = None, edge_type: Optional[list] = None, save_path: Optional[str] = None, legend: bool = True) -> NoReturn:

    # Colors
    # node_colors = mpl.cm.Pastel1(plt.Normalize(0, 8)(range(8)))
    edge_colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))

    pos = nx.kamada_kawai_layout(G)

    if node_type == None:
        map_node_type = {x: i for i, x in enumerate(set([type(node[1]['type']) for node in G.nodes.data(True)]))}

        if len(map_node_type) == 1:
            map_node_type = {x: i for i, x in enumerate(set([node[1]['type'] for node in G.nodes.data(True)]))}
            node_type = [map_node_type[node[1]['type']] for node in G.nodes.data(True)]
        else:
            node_type = [map_node_type[type(node[1]['type'])] for node in G.nodes.data(True)]

    if edge_type == None:
        map_edge_type = {x: i for i, x in enumerate(set([edge[2]['type'] for edge in G.edges.data(True)]))}
        edge_type = [map_edge_type[edge[2]['type']] for edge in G.edges.data(True)]

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    h = []

    for t, i in map_node_type.items():
        h.append(nx.draw_networkx_nodes(
            ax = ax, 
            G = G, 
            pos = pos,
            nodelist = [node for node, nt in zip(G.nodes, node_type) if nt == i],
            node_color = [i for node, nt in zip(G.nodes, node_type) if nt == i], 
            # node_size = i * 500 + 100,
            node_size = 100,
            cmap = mpl.cm.Pastel1, vmin = 0, vmax = 8,
            alpha = 0.8,
            label = t
        ))

    for t, i in map_edge_type.items():
        __ = nx.draw_networkx_edges(
            ax = ax,
            G = G,
            pos = pos,
            arrows = True,
            edgelist = [edge[:2] for edge in G.edges.data(True) if edge[2]['type'] == t], 
            label = t,
            edge_color = [i for edge in G.edges.data(True) if edge[2]['type'] == t],
            edge_cmap = mpl.cm.tab10,
            edge_vmin = 0, edge_vmax = 10,
            connectionstyle = 'arc3,rad=0.2'
        )

    __ = nx.draw_networkx_labels(
        ax = ax,
        G = G, 
        pos = pos,
        labels = {node: node for node in G.nodes},
        font_size = 8
    )

    if legend == True:
        __ = ax.legend(handles = h + [mpl.patches.FancyArrow(0, 0, 0, 0, color = edge_colors[i, :], width = 1) for t, i in map_edge_type.items()], labels = list(map_node_type.keys()) + list(map_edge_type.keys()))

    if save_path != None:
        fig.savefig(save_path, dpi = 150)

# %%
# Remove `Total_population` state

# %%


# %%
models = {}
models[info] = {}

# SBML XML as pretty string

root, dirs, files = next(os.walk(path + "/src/main"))
models[info]['SBML'] = etree.tostring(etree.parse(os.path.join(root, files[0])), pretty_print = True, encoding = str)

# MMT templates JSON
with open(path + '/model_mmt_templates.json', 'r') as f:
    models[info]['MMT'] = json.load(f)['templates']

# Petri net JSON
with open(path + f'/model_petri.json', 'r') as f:
    models[info]['Petri'] = json.load(f)


# %%
# Build graphs
for info in models.keys():
    for rep in ('MMT', 'Petri'):

        #  source-target DataFrame
        models[info][f'{rep}_df'] = build_mira_df(models[info][rep], rep = rep)

        # NetworkX graph
        models[info][f'{rep}_G'] = build_graph(models[info][f'{rep}_df'])

        # HyperNetX hypergraph
        models[info][f'{rep}_H'] = build_hypergraph(models[info][rep], rep = rep)

# %%
NUM_MODELS = 1

fig, axes = plt.subplots(nrows = 4, ncols = NUM_MODELS, figsize = (12, 16))
fig.subplots_adjust(wspace = 0.02, hspace = 0.02)

for i, info in enumerate(models.keys()):

    __ = plt.setp(axes[0], title = info)

    draw_graph(models[info]['MMT_G'], ax = axes[0], legend = False)

    hnx.draw(models[info]['MMT_H'], ax = axes[1], node_labels_kwargs = {'fontsize': 8})
    axes[1].axis('on')
    __ = axes[1].tick_params(left = True, bottom = True, labelleft = False, labelbottom = False)
    __ = plt.setp(axes[1], xticks = [], yticks = [])
    

    draw_graph(models[info]['Petri_G'], ax = axes[2], legend = False)

    hnx.draw(models[info]['Petri_H'], ax = axes[3], node_labels_kwargs = {'fontsize': 8})
    axes[3].axis('on')
    __ = axes[3].tick_params(left = True, bottom = True, labelleft = False, labelbottom = False)
    __ = plt.setp(axes[3], xticks = [], yticks = [])

__ = plt.setp(axes[0], ylabel = 'MMT as Graph')
__ = plt.setp(axes[1], ylabel = 'MMT as Hypergraph')
__ = plt.setp(axes[2], ylabel = 'Petri Net as Graph')
__ = plt.setp(axes[3], ylabel = 'Petri Net as Hypergraph')


fig.savefig(f'../figures/hackathon_20230126_{info}.png', dpi = 150)

# %%
# Load California COVID-19 dataset

df_cases = pd.read_csv('../data/hackathon_20230126/covid19cases_test.csv')
df_cases = df_cases[df_cases['area'] == 'Los Angeles']

# Total population
N = df_cases[df_cases['date'] == '2021-12-28']['population'].values[0]

# COVID-19 recovery time ~ 10 days
R = df_cases[df_cases['date'] == '2021-12-18']['cumulative_cases'].values[0]

I = df_cases[df_cases['date'] == '2021-12-28']['cumulative_cases'].values[0] - df_cases[df_cases['date'] == '2021-12-18']['cumulative_cases'].values[0] + df_cases[df_cases['date'] == '2021-12-28']['cases'].values[0]

df_hosp = pd.read_csv('../data/hackathon_20230126/covid19hospitalbycounty.csv')
df_hosp = df_hosp[df_hosp['county'] == 'Los Angeles']
df_hosp = df_hosp[df_hosp['todays_date'] == '2021-12-28']
H = df_hosp['hospitalized_covid_patients'].values[0]
# H = 1367

# Exposed ratio ~ hospitalized-suspected / hospitalized-confirmed
exposed_ratio = df_hosp['hospitalized_suspected_covid_patients'].values[0] / df_hosp['hospitalized_covid_confirmed_patients'].values[0]
E = np.round(exposed_ratio * I)

S = N - E - I - R - H

print(f'N = {N}\nS = {S}\nE = {E}\nI = {I}\nR = {R}\nH = {H}')

# N = 10257557.0
# S = 8544853.0
# E = 16865.0
# I = 181875.0
# R = 1512597.0
# H = 1367.0

# %%
# 2-month time series
t0 = df_cases[df_cases['date'] == '2021-10-18'].index.values[0]
t1 = df_cases[df_cases['date'] == '2021-10-28'].index.values[0]
t2 = df_cases[df_cases['date'] == '2021-12-18'].index.values[0]
t3 = df_cases[df_cases['date'] == '2021-12-28'].index.values[0]

N = df_cases['population'].loc[t1:t3].values
R = df_cases['cumulative_cases'].loc[t0:t2].values
I = df_cases['cumulative_cases'].loc[t1:t3].values - df_cases['cumulative_cases'].loc[t0:t2].values + df_cases['cases'].loc[t1:t3].values

df_hosp = pd.read_csv('../data/hackathon_20230126/covid19hospitalbycounty.csv')
df_hosp = df_hosp[df_hosp['county'] == 'Los Angeles']
t1_ = df_hosp[df_hosp['todays_date'] == '2021-10-28'].index.values[0]
t3_ = df_hosp[df_hosp['todays_date'] == '2021-12-28'].index.values[0]
H = df_hosp['hospitalized_covid_patients'].loc[t1_:t3_].values

E = np.round(exposed_ratio * I)

S = N - E - I - R - H

df = pd.DataFrame({'date': df_cases['date'].loc[t1:t3].values, 'S': S, 'E': E, 'I': I, 'R': R, 'H': H, 'N': N})

df.to_csv('../data/hackathon_20230126/SEIRHN.csv', index = False)

# %%
for k, v in model_mmt_parameters['parameters'].items():
    print(f"{v['name']} = {v['value']}")

# mu = 0.012048
# beta = 0.833
# epsilon = 0.33333
# gamma = 0.125
# alpha = 0.006
# City = 1.0
# XXlambdaXX = 120480.0

# Assume:
# * H ~ Threatened in SIDARTHE model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7175834/)
# * h = (Ailing -> Threatened) + (Recognized -> Threatened)
# * r = (Threatened -> Healed) + (Threatened -> Death)

# %%
