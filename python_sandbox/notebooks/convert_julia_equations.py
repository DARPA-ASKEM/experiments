# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# Build the Petri Net representation of the BIOMD0000000955 (SIDARTHE) model from scratch
# 

# %%
import os
import re

PATH = "../../notebooks/Nelson/decapodes/dome_model"

# %%
with open(os.path.join(PATH, "dome_model_de_flat_sm_vec.txt"), "r") as f:
    s_input = f.read()

# %%
ssa_map = {}
for eq in s_input.split("Equations:\n")[1].split("\n"):
    if "=" in eq:
        lhs, rhs = eq.split("=")
        lhs = lhs.strip()
        # rhs = rhs.strip()
        rhs = f" {rhs} "
        ssa_map[lhs] = rhs

# %%
import networkx as nx
import matplotlib.pyplot as plt

# %%
G = nx.DiGraph()
G.add_nodes_from(list(ssa_map.keys()))
G.add_edges_from(list(ssa_map.items()))
G.add_edges_from([(rhs, lhs) for lhs in ssa_map.keys() for rhs in ssa_map.values() if len(re.findall(f"({lhs})" + r"(?=[\D\W])(?![_])", rhs)) > 0])

for node in G.nodes:
    G.nodes[node]["expr"] = node

# %%
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
pos = nx.nx_agraph.graphviz_layout(G, prog = "dot")
# nx.draw_networkx(G, pos = pos, with_labels = True, font_size = 10)
nx.draw_networkx(G, pos = pos, labels = {node: f'{node} = {G.nodes[node]["expr"]}' for node in G.nodes}, with_labels = True, font_size = 10)
xlim = plt.getp(ax, "xlim")
ylim = plt.getp(ax, "ylim")

# %%
l = nx.dag_longest_path_length(G)

for i in range(l):

    leafs = [node for node, deg in G.out_degree if deg == 0]

    for leaf in leafs:

        neighs = list(nx.all_neighbors(G, leaf))

        if len(neighs) > 0:

            for neigh in neighs:
                leaf_expr = G.nodes[leaf]["expr"]
                neigh_expr = G.nodes[neigh]["expr"]

                p = re.compile(f"({leaf})" + r"(?=[\D\W])(?![_])")

                if len(re.findall(p, neigh_expr)) > 0:
                    G.nodes[neigh]["expr"] = re.sub(p, f"({leaf_expr.strip()})", neigh_expr)
                else:
                    G.nodes[neigh]["expr"] = leaf_expr.strip()

            G.remove_node(leaf)

# %%
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
# nx.draw_networkx(G, pos = pos, with_labels = True, font_size = 10)
nx.draw_networkx(G, pos = pos, labels = {node: f'{node} = {G.nodes[node]["expr"]}' for node in G.nodes}, with_labels = True, font_size = 10)
__ = plt.setp(ax, xlim = xlim, ylim = ylim)

# %%
with open(os.path.join(PATH, "dome_model_de_flat_sm_vec_.txt"), "w") as f:
    s = ''.join([f'{node} = {G.nodes[node]["expr"]}\n' for node in G.nodes])
    f.write(s)

# %%
