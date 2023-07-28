# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# # Experiment with ASKEM Model Representation (AMR)
#
# New DARPA-ASKEM model representation that store both the model data (Petri net) and its metadata.
# 
# Check what is available to visualize.

# %%
import uuid
import copy
import itertools
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any


# %%
# Print structure of dict
def print_dict(d: dict, l: int = 0) -> NoReturn:
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{' ' * l}{k}: <{type(v).__name__}>")
            print_dict(v, l = l + 1)
        else:
            if isinstance(v, list):
                print(f"{' ' * l}{k}: <{type(v).__name__}> of <{type(v[0]).__name__}>")
                if isinstance(v[0], dict):
                    print_dict(v[0], l = l + 1)
            else:
                print(f"{' ' * l}{k}: <{type(v).__name__}>")

# Convert amr into NX graph
def convert_amr_to_nxgraph(amr: dict) -> nx.MultiDiGraph:

    # Build NX graph
    G = nx.MultiDiGraph()

    # Add nodes
    nodes = []
    for node_type in ('states', 'transitions'):
        for node in amr['model'][node_type]:
            node = (node['id'], {**{'type': node_type}, **node})
            if 'name' not in node[1].keys():
                node[1]['name'] = node[1]['id']
                # node[1]['name'] = node[1]['properties']['rate']['expression']
            nodes.append(node)

    G.add_nodes_from(nodes)

    # Add edges
    edges = []
    for transition in amr['model']['transitions']:

        # state -> transition
        for input in transition['input']:
            edge = (input, transition['id'], {**{'type': None, 'id': str(uuid.uuid4())}, **transition['properties']})
            edges.append(edge)

        # transition -> state
        for output in transition['output']:
            edge = (transition['id'], output, {**{'type': None, 'id': str(uuid.uuid4())}, **transition['properties']})
            edges.append(edge)

    G.add_edges_from(edges)

    return G

# Draw Petri net with given AMR or NX graph
def draw_petri(amr: Optional[dict] = None, G: Optional[nx.MultiDiGraph] = None, ax: Optional[Any] = None, legend: bool = True) -> NoReturn:

    if amr != None:
        G = convert_amr_to_nxgraph(amr)

    # Edge colours
    edge_colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))

    # Layout
    nodelist = {
        node_type: [node[0] for node in G.nodes(data = 'type') if node[1] == node_type]
        for node_type in ('states', 'transitions')
    }
    edgelist = {
        edge_type: [(edge[0], edge[1], edge[2]) for edge in G.edges(keys = True, data = 'type') if edge[3] == edge_type]
        for edge_type in (None, )
    }
    # pos = nx.shell_layout(G, nlist = list(nodelist.values()))
    pos = nx.kamada_kawai_layout(G)

    # Node shape
    node_shape = {'states': 'o', 'transitions': 's'}

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        ax.set_aspect(1.0)

    h = []

    # Draw nodes
    for i, node_type in enumerate(nodelist.keys()):
        h.append(nx.draw_networkx_nodes(
            ax = ax, 
            G = G, 
            pos = pos,
            nodelist = nodelist[node_type],
            node_color = [i for __ in range(len(nodelist[node_type]))], 
            node_size = 200,
            node_shape = node_shape[node_type],
            cmap = mpl.cm.Pastel1, 
            vmin = 0, 
            vmax = 8,
            alpha = 0.8,
            label = node_type
        ))

    # Node labels
    node_labels = dict(G.nodes(data = 'name'))
    __ = nx.draw_networkx_labels(
        ax = ax,
        G = G, 
        pos = pos,
        labels = node_labels,
        font_size = 8
    )

    # Draw edges
    edge_width = {edge[2]['id']: G.number_of_edges(edge[0], edge[1]) for edge in G.edges(data = True)}
    for i, edge_type in enumerate(edgelist.keys()):
        __ = nx.draw_networkx_edges(
            ax = ax,
            G = G,
            pos = pos,
            arrows = True,
            width = [edge_width[edge[2]['id']] for edge in G.edges(data = True) if edge[2]['type'] == edge_type],
            edgelist = edgelist[edge_type], 
            label = edge_type,
            edge_color = [i for __ in range(len(edgelist[edge_type]))],
            edge_cmap = mpl.cm.tab10,
            edge_vmin = 0, 
            edge_vmax = 10,
            alpha = 0.5,
            connectionstyle = 'arc3,rad=0.2'
        )

    # Draw legend
    if legend == True:
        __ = ax.legend(
            handles = h + [
                mpl.patches.FancyArrow(0, 0, 0, 0, color = edge_colors[i, :], width = 1.0, alpha = 0.5) 
                for i, __ in enumerate(edgelist.keys())
            ], 
            labels = list(nodelist.keys()) + list(edgelist.keys()), 
            loc = 'upper center', 
            bbox_to_anchor = (0.5, -0.01), 
            ncols = 4
        )

    # Square axis limits
    m = max(np.abs(np.array(plt.getp(ax, 'xlim') + plt.getp(ax, 'ylim'))))
    __ = plt.setp(ax, xlim = (-m, m), ylim = (-m, m))

# %%
# Load the AMR of a stratified model using MIRA and Catlab
with open('../data/catlab_vs_mira/sir_loc_model.json', 'r') as f:
    amr_mira = json.load(f)

with open('../data/catlab_vs_mira/cat123123123.json', 'r') as f:
    amr_catlab = json.load(f)

# %%
G_mira = convert_amr_to_nxgraph(amr_mira)
G_catlab = convert_amr_to_nxgraph(amr_catlab)

# %%
# Rename transition nodes with expressions

rates = {r['target']: r for r in amr_mira['semantics']['ode']['rates']}
for t in amr_mira['model']['transitions']:
    expression = rates[t['id']]['expression']
    for input in t['input']:
        expression = expression.replace(input + '*', '')
        expression = expression.replace('*' + input, '')
        expression = expression.replace('*' + input + '*', '')
    nx.set_node_attributes(G_mira, {t['id']: expression}, name = 'name')


rates = {r['target']: r for r in amr_catlab['semantics']['ode']['rates']}
for t in amr_catlab['model']['transitions']:
    expression = rates[t['id']]['expression']
    for input in t['input']:
        expression = expression.replace(input + '*', '')
        expression = expression.replace('*' + input, '')
        expression = expression.replace('*' + input + '*', '')
    nx.set_node_attributes(G_catlab, {t['id']: expression}, name = 'name')


# %%
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 12))
draw_petri(G = G_mira, legend = True, ax = axes[0])
draw_petri(G = G_catlab, legend = True, ax = axes[1])

__ = plt.setp(axes[0], title = 'MIRA')
__ = plt.setp(axes[1], title = 'Catlab stratification & MIRA Reconstruction')

# %%
