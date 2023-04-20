# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # Collapsed Petri Net Visualization
#
# Implement "fast" or "naive" model stratification and collapsed layout.

# %%
import itertools
import copy
import uuid
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import circlify as circ
# import squarify as sqrf
from typing import NoReturn, Optional, Any

# %%
# Example of a base model as a Petri net

acset_base = {
    "S": [
        {"sname": "I"},
        {"sname": "R"},
        {"sname": "S"}
    ],
    "T": [
        {"tname": "β"},
        {"tname": "γ"}
    ],
    "I": [
        {"it": 1, "is": 1},
        {"it": 1, "is": 3},
        {"it": 2, "is": 1}
    ],
    "O": [
        {"ot": 1, "os": 1},
        {"ot": 1, "os": 1},
        {"ot": 2, "os": 2}
    ]
}

# %%
# Convert ACSet into NX graph
def convert_acset_to_nxgraph(acset: dict) -> nx.MultiDiGraph:

    # Generate UUID for ACSet nodes and edges
    acset_uuid = copy.deepcopy(acset)
    for k in acset_uuid.keys():
        for s in acset_uuid[k]:
            s['uuid'] = str(uuid.uuid4())

    # Map S, T indices to UUID
    map_ind_uuid = {
        'S': {(i + 1): s['uuid'] for i, s in enumerate(acset_uuid['S'])}, 
        'T': {(j + 1): s['uuid'] for j, s in enumerate(acset_uuid['T'])},
    }

    # Build NX graph
    G = nx.MultiDiGraph()

    # Add nodes
    for node_type in ('S', 'T'):
        G.add_nodes_from([
            (node['uuid'], {'type': node_type, 'name': node[node_type.lower() + 'name'], 'uuid': node['uuid']}) for node in acset_uuid[node_type]
        ])

    # # Add edges
    for edge_type, (src, tgt) in zip(('I', 'O'), (('is', 'it'), ('ot', 'os'))):
        G.add_edges_from([
            (map_ind_uuid[src[1].upper()][edge[src]], map_ind_uuid[tgt[1].upper()][edge[tgt]], {'type': edge_type, 'uuid': edge['uuid']}) 
            for edge in acset_uuid[edge_type]
        ])

    return acset_uuid, G

# Convert NX graph into ACSet
def convert_nxgraph_to_acset(G: nx.MultiDiGraph) -> dict:

    acset = {}

    # Build node lists
    for node_type in ('S', 'T'):
        acset[node_type] = [
            {
                node_type.lower() + 'name': node[1]['name'],
                # 'uuid': node[1]['uuid'],
            } 
            for node in G.nodes(data = True) if node[1]['type'] == node_type
        ]

    # Map node UUID to index
    map_node_uuid_ind = {
        node_type: {node[1]['uuid']: -1 for node in G.nodes(data = True) if node[1]['type'] == node_type}
        for node_type in ('S', 'T')
    }
    for node_type in ('S', 'T'):
        for i, k in enumerate(map_node_uuid_ind[node_type].keys()):
            map_node_uuid_ind[node_type][k] = i + 1

    # Build edge lists
    for edge_type, (src, tgt) in zip(('I', 'O'), (('is', 'it'), ('ot', 'os'))):
        acset[edge_type] = [
            {
               src: map_node_uuid_ind[src[1].upper()][edge[0]],
               tgt: map_node_uuid_ind[tgt[1].upper()][edge[1]],
               # 'uuid': edge[2]['uuid'],
            }
            for edge in G.edges(data = True) if edge[2]['type'] == edge_type
        ]

    return acset

# Draw Petri net with given ACSet or NX graph
def draw_petri(acset: Optional[dict] = None, G: Optional[nx.MultiDiGraph] = None, ax: Optional[Any] = None, legend: bool = True) -> NoReturn:

    if acset != None:
        __, G = convert_acset_to_nxgraph(acset)

    # Edge colours
    edge_colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))

    # Layout
    nodelist = {
        node_type: [node[0] for node in G.nodes(data = 'type') if node[1] == node_type]
        for node_type in ('S', 'T')
    }
    edgelist = {
        edge_type: [(edge[0], edge[1]) for edge in G.edges(data = 'type') if edge[2] == edge_type]
        for edge_type in ('I', 'O')
    }
    # pos = nx.shell_layout(G, nlist = list(nodelist.values()))
    pos = nx.kamada_kawai_layout(G)

    # Node shape
    node_shape = {'S': 'o', 'T': 's'}

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
    __ = nx.draw_networkx_labels(
        ax = ax,
        G = G, 
        pos = pos,
        labels = dict(G.nodes(data = 'name')),
        font_size = 8
    )

    # Draw edges
    edge_width = {edge[2]['uuid']: G.number_of_edges(edge[0], edge[1]) for edge in G.edges(data = True)}
    for i, edge_type in enumerate(edgelist.keys()):
        __ = nx.draw_networkx_edges(
            ax = ax,
            G = G,
            pos = pos,
            arrows = True,
            width = [edge_width[edge[2]['uuid']] for edge in G.edges(data = True) if edge[2]['type'] == edge_type],
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

# Do "naive" stratification
# 1. replicate each state node n_strat times
# 2. replicate each transition node n_strat ** 2 times
# 3. create intra-strata transitions (n_strat ** 2) and edges
# 4. replicate state-transition edges for all combinations strata state nodes
def naive_stratify(acset: Optional[dict] = None, G_base: Optional[nx.MultiDiGraph] = None, n_strat: Optional[int] = 2) -> dict:

    # Convert base ACSet into NX graph
    if acset != None:
        __, G_base = convert_acset_to_nxgraph(acset)

    # Build stratified NX graph
    G_strat = nx.MultiDiGraph()

    # For every state node in G_base, 
    # 1. create n_strat child nodes in G_strat
    # 2. for each pair of child nodes, create a 1-to-1 transition nodes and edges in G_strat
    # Parents of the transition nodes and edges is added to the base graph

    # for every state node in G_base
    nodes = []
    edges = []
    nodes_base = []
    edges_base = []
    for node in G_base.nodes(data = True):
        if node[1]['type'] == 'S':

            # create n_strat child nodes for each state node in G_base
            ids = [str(uuid.uuid4()) for i in range(n_strat)]
            children = [
                (
                    id, 
                    {
                        'type': node[1]['type'], 
                        'name': f"{node[1]['name']}_{str(i)}",
                        'uuid': id,
                        'parent_uuid': node[1]['uuid'],
                        'process': 0
                    }
                )
                for i, id in zip(range(n_strat), ids)
            ]
            
            nodes.extend(children)

            # 1-to-1 interaction between every pair of child nodes (including self-loops)
            id_tnode_parent = str(uuid.uuid4())
            id_edge_I = str(uuid.uuid4())
            id_edge_O = str(uuid.uuid4())
            for (child_0, child_1) in itertools.product(children, repeat = 2):
                
                # Transition node
                id = str(uuid.uuid4())
                nodes.append(
                    (
                        id,
                        {
                            'type': 'T',
                            'name': f"{node[1]['name']}{node[1]['name']}_({child_0[1]['name']}, {child_1[1]['name']})",
                            'uuid': id,
                            'parent_uuid': id_tnode_parent,
                            'process': 2
                        }
                    )
                )

                # S -> T, T -> S edges
                edges.extend(
                    [
                        (
                            child_0[0],
                            nodes[-1][0],
                            {
                                'type': 'I',
                                'name': None,
                                'uuid': str(uuid.uuid4()),
                                'parent_uuid': id_edge_I
                            }
                        ),
                        (
                            nodes[-1][0],
                            child_1[0],
                            {
                                'type': 'O',
                                'name': None,
                                'uuid': str(uuid.uuid4()),
                                'parent_uuid': id_edge_O
                            }
                        )
                    ]    
                )

            # Add the parents to G_base
            nodes_base.append(
                    (
                        id_tnode_parent,
                        {
                            'type': 'T',
                            'name': f"{node[1]['name']}{node[1]['name']}",
                            'uuid': id_tnode_parent,
                            'parent_uuid': None
                        }
                    )
            )
            # S -> T, T -> S edges
            edges_base.extend(
                [
                    (
                        node[1]['uuid'],
                        id_tnode_parent,
                        {
                            'type': 'I',
                            'name': None,
                            'uuid': str(uuid.uuid4()),
                            'parent_uuid': id_edge_I
                        }
                    ),
                    (
                        id_tnode_parent,
                        node[1]['uuid'],
                        {
                            'type': 'O',
                            'name': None,
                            'uuid': str(uuid.uuid4()),
                            'parent_uuid': id_edge_O
                        }
                    )
                ]    
            )

    G_strat.add_nodes_from(nodes)
    G_strat.add_edges_from(edges)


    # For every transition node in G_base,
    # 1. find its incident edges in G_base
    # 2. find its neighbouring state nodes in G_base
    # 3. compute all possible combinations of the children of these neighbouring state nodes in G_strat using itertools(range(n_strat), repeat = n_neigh)
    # 4. create a child transition node for each combination
    # 5. for every one of its incident edges, ...

    # Map between parent and child state node UUIDs
    map_child_parent = dict(G_strat.nodes(data = 'parent_uuid'))
    map_parent_child = {parent: [c for c, p in map_child_parent.items() if p == parent] for parent in set(dict(G_strat.nodes(data = 'parent_uuid')).values())}
    
    # for every transition node in G_base
    nodes = []
    edges = []
    for node in G_base.nodes(data = True):
        if node[1]['type'] == 'T':

            # find its neighbouring state nodes in G_base
            neigh_nodes = list(set(G_base.predecessors(node[0])) | set(G_base.successors(node[0])))
            n_neigh = len(neigh_nodes)

            # compute combinations of neighbours
            # for every combination, create a child transition node
            neigh_combi = itertools.product(range(n_strat), repeat = n_neigh)
            neigh_children = [[child for child in map_parent_child[neigh_node] if G_strat.nodes[child]['type'] == 'S'] for neigh_node in neigh_nodes]
            for combi in neigh_combi:
                
                # get the UUID of the child state nodes of the neighbouring state nodes
                combi_nodes_uuid = [neigh_children[ind_neigh][ind_neigh_child] for ind_neigh, ind_neigh_child in enumerate(combi)]

                # create a child transition node
                id = str(uuid.uuid4())
                child = (
                    id,
                    {
                        'type': 'T',
                        'name': f"{node[1]['name']}_({', '.join([G_strat.nodes[n]['name'] for n in combi_nodes_uuid])})",
                        'uuid': id,
                        'parent_uuid': node[1]['uuid'],
                        'process': 1
                    }
                )
                nodes.append(child)

                # for each neighbouring state node, 
                # find every edge between this transition node and that state node in G_base
                # replicate this edge in G_strat
                for neigh_node_uuid in combi_nodes_uuid:
                    
                    # UUID of the parents in G_base
                    parent_S_uuid = G_strat.nodes[neigh_node_uuid]['parent_uuid']
                    parent_T_uuid = child[1]['parent_uuid']
                    m = {parent_S_uuid: neigh_node_uuid, parent_T_uuid: child[1]['uuid']}

                    # find the incident edges in G_base (with keys to deal with multiple edges)
                    edges_b = list((set(G_base.in_edges(parent_T_uuid, keys = True)) | set(G_base.out_edges(parent_T_uuid, keys = True))) & (set(G_base.in_edges(parent_S_uuid, keys = True)) | set(G_base.out_edges(parent_S_uuid, keys = True))))
                    
                    # create edge in G_strat for every such edge
                    edges.extend([
                        (
                            m[edge[0]],
                            m[edge[1]],
                            {
                                'type': G_base.edges[edge]['type'],
                                'name': None,
                                'uuid': str(uuid.uuid4()),
                                'parent_uuid': G_base.edges[edge]['uuid']
                            }
                        )
                        for edge in edges_b
                    ])
         
    G_strat.add_nodes_from(nodes)
    G_strat.add_edges_from(edges)

    G_base.add_nodes_from(nodes_base)
    G_base.add_edges_from(edges_base)

    # Convert NX graph to ACSet
    acset_strat = convert_nxgraph_to_acset(G_strat)

    return acset_strat, G_strat, G_base

# De-stratify a stratified NX graph to get the original base NX graph
# Recreate the base nodes and edges using 1st child's metadata
def naive_destratify(G_strat: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int]:

    # Initialize
    G_destrat = nx.MultiDiGraph()
    int = 0

    # Check if G_strat was stratified
    node = next(iter(G_strat.nodes(data = True)))
    if 'parent_uuid' in node[1].keys():

        # Map between parent and child node UUIDs
        map_child_parent = dict(G_strat.nodes(data = 'parent_uuid'))
        map_parent_child = {parent: [c for c, p in map_child_parent.items() if p == parent] for parent in set(dict(G_strat.nodes(data = 'parent_uuid')).values())}
        
        # Recreate parent nodes with 1st child's metadata
        n_destrat = [len(children) for parent, children in map_parent_child.items() if G_strat.nodes[children[0]]['type'] == 'S'][0]
        nodes = []
        for parent, children in map_parent_child.items():
            if parent != None:
                child = children[0]
                nodes.append((
                    parent,
                    {   
                        'type': G_strat.nodes[child]['type'],
                        'name': G_strat.nodes[child]['name'].split('_')[0],
                        'uuid': parent,
                        'parent_uuid': None,
                        'children_uuid': children
                    }
                ))

        G_destrat.add_nodes_from(nodes)

        # Recreate parent edges with 1st child's metadata
        edges = []
        map_edges_child_parent = {edge[3]['uuid']: edge[3]['parent_uuid'] for edge in G_strat.edges(data = True, keys = True)}
        map_edges_parent_child = {parent: [edge[3]['uuid'] for edge in G_strat.edges(data = True, keys = True) if edge[3]['parent_uuid'] == parent] for parent in set(map_edges_child_parent.values())}
        map_edges_child_metadata = {edge[3]['uuid']: edge for edge in G_strat.edges(data = True, keys = True)}
        for parent, children in map_edges_parent_child.items():
            if parent != None:
                child = children[0]
                edges.append((
                    map_child_parent[map_edges_child_metadata[child][0]],
                    map_child_parent[map_edges_child_metadata[child][1]],
                    map_edges_child_metadata[child][2],
                    {
                        'type': map_edges_child_metadata[child][3]['type'],
                        'uuid': parent,
                        'parent_uuid': None,
                        'children_uuid': children
                    }
                ))
        
        G_destrat.add_edges_from(edges)

    else:
        print(f'Error: Missing `parent_uuid` node attribute in `G_strat`.')

    return (G_destrat, n_destrat)

# %%
# Draw stratified Petri net in a collapsed layout
# 1. destratify to get the parent graph
# 2. compute parent graph layout
# 3. compute circle-packing layout of each children/strata subgraph
# 4. 
def draw_petri_collapsed(G_strat: Optional[nx.MultiDiGraph] = None, ax: Optional[Any] = None, legend: bool = True) -> NoReturn:

    # De-stratify `G` to get the root graph
    G_parent, __ = naive_destratify(G_strat)

    # Layout of parent graph
    pos_parent = nx.kamada_kawai_layout(G_parent, center = (0, 0), dim = 2)
    rad_parent = {id_parent: 0.1 for id_parent in pos_parent.keys()}

    # For each parent,
    # 1. find its children
    # 2. compute a circle-pack layout for this subgraph
    # 3. shift to parent position and rescale to fit inside parent node
    pos_child = {}
    rad_child = {}
    for node in G_parent.nodes(data = True):

        # Circle-pack layout of each parent node's children nodes
        data = [
            {
                'id': id_child,
                'datum': 1.0,
                'children': []

            } 
            for id_child in node[1]['children_uuid']
        ]
        circles = circ.circlify(
            data, 
            show_enclosure = False, 
            target_enclosure = circ.Circle(x = pos_parent[node[0]][0], y = pos_parent[node[0]][1], r = rad_parent[node[0]])
        )

        # Accumulate for all children nodes
        pos_child = {**pos_child, **{id_child: np.array([x, y]) for id_child, (x, y, r) in zip(node[1]['children_uuid'], circles)}}
        rad_child = {**rad_child, **{id_child: r for id_child, (x, y, r) in zip(node[1]['children_uuid'], circles)}}

    # Setup canvas
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        ax.set_aspect(1.0)

    # Colors
    # colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))
    colors = mpl.cm.Pastel1(plt.Normalize(0, 8)(range(8)))
    map_type_color = {'S': colors[0], 'T': colors[1]}

    # Draw parent nodes
    for id_parent in G_parent.nodes:
        x, y = pos_parent[id_parent]
        r = rad_parent[id_parent]
        circle = mpl.patches.Circle(
            (x, y), 
            r, 
            facecolor = map_type_color[G_parent.nodes[id_parent]['type']], 
            edgecolor = 'k', 
            linewidth = 0.5, 
            alpha = 0.5
        )
        __ = ax.add_patch(circle)
        __ = ax.text(x, y + r, G_parent.nodes[id_parent]['name'], fontsize = 'medium', color = map_type_color[G_parent.nodes[id_parent]['type']], ha = 'center', va = 'bottom')

    # Draw parent edges
    for edge in G_parent.edges:
        src = pos_parent[edge[0]]
        tgt = pos_parent[edge[1]]
        w = G_parent.number_of_edges(edge[0], edge[1])
        __ = ax.annotate(
            '', 
            xy = src, 
            xytext = tgt, 
            xycoords = 'data', 
            color = 'k',
            arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3, rad = 0.2', 'facecolor': colors[3], 'edgecolor': None, 'linewidth': 0.5 * w, 'shrinkA': 10, 'shrinkB': 10}
        )

    # Draw child nodes
    for id_child in G_strat.nodes:
        x, y = pos_child[id_child]
        r = rad_child[id_child]
        circle = mpl.patches.Circle(
            (x, y), 
            r, 
            facecolor = map_type_color[G_strat.nodes[id_child]['type']], 
            edgecolor = 'k', 
            linewidth = 0.5, 
            alpha = 0.5
        )
        __ = ax.add_patch(circle)
        # __ = ax.text(x, y, '_'.join(G_strat.nodes[id_child]['name'].split('_')[1:]), fontsize = 'xx-small', color = map_type_color[G_strat.nodes[id_child]['type']], ha = 'center', va = 'center')


    # Square axis limits
    m = 1.2 * max(np.abs(np.array(plt.getp(ax, 'xlim') + plt.getp(ax, 'ylim'))))
    __ = plt.setp(ax, xlim = (-m, m), ylim = (-m, m))

    ax.tick_params('both', left = False, bottom = False, labelleft = False, labelbottom = False)

    # Draw legend
    # if legend == True:
    #     __ = ax.legend(
    #         handles = h
    #         labels = list(nodelist.keys()) + list(edgelist.keys()), 
    #         loc = 'upper center', 
    #         bbox_to_anchor = (0.5, -0.01), 
    #         ncols = 4
    #     )

# %%
# Test conversion between ACSet and NX graph
acset_base_uuid, G_base = convert_acset_to_nxgraph(acset_base)

# Naive stratification
n_strat = 2
acset_strat, G_strat, G_base_ = naive_stratify(acset_base, n_strat = n_strat)

# %%
fig, axes = plt.subplots(1, 3, figsize = (18, 6))
fig.subplots_adjust(wspace = 0.05)
__ = plt.setp(fig.axes[0], title = 'Base Model', aspect = 1.0)
__ = plt.setp(fig.axes[1], title = 'Base Model + Self-Loops', aspect = 1.0)
__ = plt.setp(fig.axes[2], title = f'Stratified Model ({n_strat} Strata)', aspect = 1.0)
draw_petri(acset = acset_base, ax = fig.axes[0], legend = True)
draw_petri(G = G_base_, ax = fig.axes[1], legend = False)
draw_petri(acset = acset_strat, ax = fig.axes[2], legend = False)

# %%
G_destrat, n_destrat = naive_destratify(G_strat)

# %%
fig, axes = plt.subplots(2, 2, figsize = (10, 10))
fig.subplots_adjust(wspace = 0.1, hspace = 0.1)
__ = plt.setp(fig.axes[0], title = f'Base Model', aspect = 1.0)
__ = plt.setp(fig.axes[1], title = f'Destratified Model ({n_destrat} strata)', aspect = 1.0)
__ = plt.setp(fig.axes[2], title = f'Stratified Model (Expanded Layout)', aspect = 1.0)
__ = plt.setp(fig.axes[3], title = f'Stratified Model (Collapsed Layout)', aspect = 1.0)
draw_petri(G = G_base, ax = fig.axes[0], legend = False)
draw_petri(G = G_destrat, ax = fig.axes[1], legend = False)
draw_petri(G = G_strat, ax = fig.axes[2], legend = False)
draw_petri_collapsed(G_strat = G_strat, ax = fig.axes[3], legend = False)

# %%
# Compare full and collapsed layouts


