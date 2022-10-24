# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# - Example of a model in meta-model-template (MMT) representation
# - Provided by Ben Gyori at HMS
# - From [EMBL BioModels](https://www.ebi.ac.uk/biomodels/BIOMD0000000955)

# %%
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any

# %%
# Get model in meta-model-template format
res = requests.get('http://34.230.33.149:8771/api/biomodels/BIOMD0000000955')

# Model in MMT representation and JSON format
model_mmt = res.json()['templates']

# %%
# Request conversion to Petri net
res = requests.post('http://34.230.33.149:8771/api/to_petrinet', json = res.json())

model_petrinet = res.json()

# S = species
# T = transition

# %%
# Build DataFrame

x = {
    'source': [],
    'source_type': [], 
    'target': [],
    'target_type': [],
    'edge_type': []
}

i = 0
for l in model_mmt:

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

df_mmt = pd.DataFrame(x)

df_mmt

# %%
# Build source-target DataFrame

map_petrinet_names = {'S': {(i + 1): s['sname'] for i, s in enumerate(model_petrinet['S'])}, 'T': {(j + 1): s['tname'] for j, s in enumerate(model_petrinet['T'])}}

df_petrinet = pd.DataFrame({
    'source': [map_petrinet_names['S'][d['is']] for d in model_petrinet['I']] + [map_petrinet_names['T'][d['ot']] for d in model_petrinet['O']],
    'source_type': ['S' for d in model_petrinet['I']] + ['T' for d in model_petrinet['O']],
    'target': [map_petrinet_names['T'][d['it']] for d in model_petrinet['I']] + [map_petrinet_names['S'][d['os']] for d in model_petrinet['O']],
    'target_type': ['T' for d in model_petrinet['I']] + ['S' for d in model_petrinet['O']],
    'edge_type': ['I' for d in model_petrinet['I']] + ['O' for d in model_petrinet['O']]
})

df_petrinet

# %%
# Build NetworkX graph
def build_graph(df: pd.DataFrame) -> nx.DiGraph:

    G = nx.DiGraph()
    for i in ['source', 'target']:
        G.add_nodes_from([(node, {'type': node_type}) for __, (node, node_type) in df[[i, f"{i}_type"]].iterrows()])

    G.add_edges_from([(edge['source'], edge['target'], {'type': edge['edge_type']}) for __, edge in df.iterrows()])
    return G

# %%
# Draw graph
def draw_graph(G: nx.DiGraph, ax: Optional[Any] = None, node_type: Optional[list] = None, edge_type: Optional[list] = None, save_path: Optional[str] = None) -> NoReturn:

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
            node_size = i * 500 + 100,
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

    __ = ax.legend(handles = h + [mpl.patches.FancyArrow(0, 0, 0, 0, color = edge_colors[i, :], width = 1) for t, i in map_edge_type.items()], labels = list(map_node_type.keys()) + list(map_edge_type.keys()))

    if save_path != None:
        fig.savefig(save_path, dpi = 150)


# %%
# Draw graphs
G_mmt = build_graph(df_mmt)
G_petrinet = build_graph(df_petrinet)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
fig.subplots_adjust(wspace = 0.01)

draw_graph(G_mmt, ax = axes[0])
__ = plt.setp(axes[0], title = 'MMT Representation')

draw_graph(G_petrinet, ax = axes[1])
__ = plt.setp(axes[1], title = 'Petri Net Representation')

fig.suptitle('Model at TA1-TA2 Integration Point')

fig.savefig('../figures/example_mmt_petrinet_graphs.png', dpi = 150)

# %%
