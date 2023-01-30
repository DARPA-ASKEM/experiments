# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# Experiment with visualization of the different model representations
# 
# Model representations:
# * Source code -> GroMEt -> Bilayer -> MMT or Petri net
# * SBML -> MMT -> Petri net
#
# Visualization representations
# * Code
# * Hypergraph rubber bands
# * Node-link graph

# %%
import os
import json
import lxml.etree as etree
import pandas as pd
import numpy as np
import networkx as nx
import hypernetx as hnx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any

# %%
NUM_MODELS = 3

# %%
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
# Load models under the different representations

model_list = pd.read_csv('../../thin-thread-examples/models.csv')

models = {}

for i, (__, __, info) in model_list.iterrows():

    if i < NUM_MODELS:

        path = f'../../thin-thread-examples/biomodels/{info}'
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

        for rep in ('MMT', 'Petri'):

            #  source-target DataFrame
            models[info][f'{rep}_df'] = build_mira_df(models[info][rep], rep = rep)

            # NetworkX graph
            models[info][f'{rep}_G'] = build_graph(models[info][f'{rep}_df'])

            # HyperNetX hypergraph
            models[info][f'{rep}_H'] = build_hypergraph(models[info][rep], rep = rep)
        

# %%
# Draw representations

fig, axes = plt.subplots(nrows = 4, ncols = NUM_MODELS, figsize = (12, 16))
fig.subplots_adjust(wspace = 0.02, hspace = 0.02)

for i, info in enumerate(models.keys()):

    __ = plt.setp(axes[0, i], title = info)

    draw_graph(models[info]['MMT_G'], ax = axes[0, i], legend = False)

    hnx.draw(models[info]['MMT_H'], ax = axes[1, i], node_labels_kwargs = {'fontsize': 8})
    axes[1, i].axis('on')
    __ = axes[1, i].tick_params(left = True, bottom = True, labelleft = False, labelbottom = False)
    __ = plt.setp(axes[1, i], xticks = [], yticks = [])
    

    draw_graph(models[info]['Petri_G'], ax = axes[2, i], legend = False)

    hnx.draw(models[info]['Petri_H'], ax = axes[3, i], node_labels_kwargs = {'fontsize': 8})
    axes[3, i].axis('on')
    __ = axes[3, i].tick_params(left = True, bottom = True, labelleft = False, labelbottom = False)
    __ = plt.setp(axes[3, i], xticks = [], yticks = [])

__ = plt.setp(axes[0, 0], ylabel = 'MMT as Graph')
__ = plt.setp(axes[1, 0], ylabel = 'MMT as Hypergraph')
__ = plt.setp(axes[2, 0], ylabel = 'Petri Net as Graph')
__ = plt.setp(axes[3, 0], ylabel = 'Petri Net as Hypergraph')

fig.savefig('../figures/viz_model_representations.png', dpi = 150)

# %%

# model_list = pd.read_csv('../../thin-thread-examples/models.csv')

# models = {}

# for i, (__, __, info) in model_list.iterrows():

#     path = f'../../thin-thread-examples/biomodels/{info}'
#     models[info] = {}

#     # MMT templates JSON
#     if os.path.exists(path + '/model_mmt_templates.json') == True:
#         with open(path + '/model_mmt_templates.json', 'r') as f:
#             models[info]['MMT'] = json.load(f)['templates']



# states = []
# for model in models.values():
#     if 'MMT' in model.keys():
#         states += [t['subject'] for t in model['MMT'] if 'subject' in t.keys()]
#         states += [t['outcome'] for t in model['MMT'] if 'outcome' in t.keys()]
#         states += [c for t in model['MMT'] if 'controllers' in t.keys() for c in t['controllers']]

# states_ = {}
# for s in states:
#     __ = s['identifiers'].pop('biomodels.species', None)
#     if len(s['identifiers']) > 0:
#         states_[json.dumps(s, sort_keys = True, default = repr, ensure_ascii = True).encode('utf-8')] = None

# list(states_.keys())

# %%
