# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Generate a model from a "model diagram"
#
# 1. Draw a (NetworkX) graph with state variables as nodes and dependencies as edges
# 2. Generate mass-action rate laws on the edges
# 3. Optionally change the rate laws to custom ones (e.g. factoring the parameters, adding "controllers" or other state variables)
# 4. 


# %%
import html
import os
import json
import requests
import uuid
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any
import latex2mathml.converter

# %%
REST_URL_SKEMA = "https://skema-rs.staging.terarium.ai"

# %%
# Populate the rate laws of the model diagram graph with mass-action kinetics
def generate_rate_laws(G: nx.DiGraph) -> nx.DiGraph:

    # number of edges = number of parameters
    num_edges = len(G.edges)
    params = [f"c_{i}" for i in range(num_edges)]
    # latin_letters = [chr(c) for c in range(97, 122)]
    # greek_letters = [chr(c) for c in range(945, 970)]
    # params = [latin_letters[i] for i in range(num_edges)]
    # params = [greek_letters[i] for i in range(num_edges)]

    for (src, tgt), p in zip(G.edges, params):

        rate_law = f"{p}"
        if src != "None":
            # rate_law += r" \operatorname{" + src + r"}"
            rate_law += r" " + src

        G.edges[(src, tgt)]["rate_law"] = rate_law

    return G

# Plot model diagram
def draw_model_diagram(G: nx.DiGraph, ax: Optional[Any] = None) -> NoReturn:

    # Colours
    # colors = mpl.cm.Pastel1(plt.Normalize(0, 8)(range(8)))
    colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))

    # Graph layout
    pos = nx.circular_layout(G, center = (0, 0))
    # pos = nx.kamada_kawai_layout(G)

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    h = []

    h.append(nx.draw_networkx_nodes(
        ax = ax,
        G = G,
        pos = pos,
        node_size = 200,
        cmap = mpl.cm.tab10, vmin = 0, vmax = 10,
        # cmap = mpl.cm.Pastel1, vmin = 0, vmax = 8,
        alpha = 0.2,
        label = "State Variables"
    ))

    h.append(nx.draw_networkx_labels(
        ax = ax,       
        G = G,
        pos = pos,
        labels = {node: node for node in G.nodes},
        font_color = colors[0],
        font_size = 8,
    ))

    h.append(nx.draw_networkx_edges(
        ax = ax,
        G = G,
        pos = pos,
        label = "Dependencies",
        edge_color = colors[1],
        arrows = True,
        arrowsize = 20,
        connectionstyle = "arc3,rad=0.2"
    ))

    h.append(nx.draw_networkx_edge_labels(
        ax = ax,
        G = G,
        pos = pos,
        edge_labels = {(src, tgt): r"$" + data["rate_law"] + r"$" for src, tgt, data in G.edges(data = True)},
        horizontalalignment = "center",
        verticalalignment = "center",
        rotate = False,
        font_color = colors[1],
        font_size = 8
    ))

# Extract equations from model diagram
# One equation for each variable node
def extract_equations(G: nx.DiGraph) -> list:

    equations = []
    for var in list(G.nodes):
        if var != "None":
            # equation = r"\frac{d \operatorname{" + var + r"}}{d t} = "
            equation = r"\frac{d " + var + r"}{d t} = "

            in_equations = " + ".join([G.edges[edge]["rate_law"] for edge in G.in_edges(var)])
            out_equations = " - ".join([G.edges[edge]["rate_law"] for edge in G.out_edges(var)])

            equation += in_equations
            if len(G.out_edges(var)) > 0:
                equation += ' - ' + out_equations

            equations.append(equation)

    return equations

# Convert LaTeX equations to MathML equations
def convert_latex2mathml(model_latex: Optional[list]) -> list:

    model_mathml = []
    for i in range(len(model_latex)):

        m = latex2mathml.converter.convert(model_latex[i])

        # Replace some unicode characters
        # SKEMA throws error otherwise
        m = m.replace("&#x0003D;", r"=") 
        m = m.replace("&#x02212;", r"-")
        m = m.replace("&#x0002B;", r"+")
        m = html.unescape(m)

        m = m.replace(r'<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">', r'<math>')

        # SKEMA requires the removal of the outer `<mrow></mrow>`
        m = m.replace(r"<math><mrow>", r"<math>")
        m = m.replace(r"</mrow></math>", r"</math>")

        model_mathml.append(m)
        
    return model_mathml

# Convert Python request into a cURL command
def request_to_curl(req: Any) -> str:

    command = "curl -X {method} '{uri}' -H {headers} -d '{data}' "
    method = req.method
    uri = req.url
    data = req.body
    headers = ['"{0}: {1}"'.format(k, v) for k, v in req.headers.items()]
    headers = " -H ".join(headers)

    return command.format(method=method, headers=headers, data=data, uri=uri)

# Convert MathML equations to Petri-net AMR
def convert_mathml2amr(model_mathml: Optional[list] = None) -> dict:

    model_amr = {}

    # Test MathML
    if model_mathml == 'test':
        model_mathml = [
            "<math><mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mo>-</mo><mi>b</mi><mi>S</mi><mi>I</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>I</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>b</mi><mi>S</mi><mi>I</mi><mo>-</mo><mi>g</mi><mi>I</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>R</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>g</mi><mi>I</mi></math>"
        ]

    # SKEMA API health check
    if model_mathml == None:
        url = f'{REST_URL_SKEMA}/ping'
        res = requests.get(url)
        print(f'{res.url}: Code {res.status_code}\n\t\"{res.text}\"')

    elif model_mathml == 'test':
        url = f'{REST_URL_SKEMA}/mathml/amr'
        res = requests.put(url, json = {"mathml": model_mathml, "model": "petrinet"})
        print(f'{res.url}: Code {res.status_code}\n\t\"{res.text}\"')
        if res.status_code == 200:
            model_amr = res.json()

    # SKEMA API for MathML-to-ACSet conversion
    else:
        url = f'{REST_URL_SKEMA}/mathml/amr'
        res = requests.put(url, json = {"mathml": model_mathml, "model": "petrinet"})
        print(f'{res.url}: Code {res.status_code}\n\t\"{res.text}\"')
        if res.status_code == 200:
            model_amr = res.json()

    return model_amr

# Convert amr into NX graph
def convert_amr_to_nxgraph(amr: dict) -> nx.MultiDiGraph:

    # Build NX graph
    G = nx.MultiDiGraph()

    if 'semantics' in amr.keys():
        map_transition_rate = {
            rate['target']: rate['expression']
            for rate in amr['semantics']['ode']['rates']
        }

    # Add nodes
    nodes = []
    for node_type in ('states', 'transitions'):
        for node in amr['model'][node_type]:
            node = (node['id'], {**{'type': node_type}, **node})
            if 'name' not in node[1].keys():
                node[1]['name'] = node[1]['id']
                # node[1]['name'] = node[1]['properties']['rate']['expression']

            if (node_type == 'transitions') and ('semantics' in amr.keys()):
                node[1]['rate_expression'] = map_transition_rate[node[1]['id']]

            nodes.append(node)

    G.add_nodes_from(nodes)

    # Add edges
    edges = []
    for transition in amr['model']['transitions']:

        # state -> transition
        for input in transition['input']:
            if "properties" in transition.keys():
                edge = (input, transition['id'], {**{'type': None, 'id': str(uuid.uuid4())}, **transition['properties']})
            else:
                edge = (input, transition['id'], {**{'type': None, 'id': str(uuid.uuid4())}})
            edges.append(edge)

        # transition -> state
        for output in transition['output']:
            if "properties" in transition.keys():
                edge = (transition['id'], output, {**{'type': None, 'id': str(uuid.uuid4())}, **transition['properties']})
            else:
                edge = (transition['id'], output, {**{'type': None, 'id': str(uuid.uuid4())}})
            edges.append(edge)

    G.add_edges_from(edges)

    return G

# Draw Petri net with given AMR or NX graph
def draw_petri(amr: Optional[dict] = None, G: Optional[nx.MultiDiGraph] = None, ax: Optional[Any] = None, legend: bool = True, layout: Optional[str] = None) -> NoReturn:

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
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)

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
    if "semantics" in amr.keys():
        node_labels = {
            node: data['rate_expression'] if 'rate_expression' in data.keys() else data['name']
            for node, data in G.nodes(data = True)
        }
    else:
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
# Draw a model diagram as a NetworkX graph
G = nx.DiGraph()

# S -> E
G.add_nodes_from(["S", "E"])
G.add_edge("S", "E", rate_law = "")

# E -> I
G.add_node("I")
G.add_edge("E", "I", rate_law = "")

# I -> R
G.add_node("R")
G.add_edge("I", "R", rate_law = "")

# I -> None (death)
G.add_node("None")
G.add_edge("I", "None", rate_law = "")

# None -> S (birth)
G.add_edge("None", "S", rate_law = "")

# Generate mass-action rate laws
G = generate_rate_laws(G)

# %%
# Draw model diagram
fig, axes = plt.subplots(2, 3, figsize = (12, 8))
draw_model_diagram(G, ax = axes[0, 0])

equations = extract_equations(G)
__ = plt.setp(axes[0, 0], title = '1. Construct SEIR Model as Diagram')

__ = [axes[1, 0].text(0.05, i / len(equations) + 0.5 / len(equations), s = "$" + equation + "$", va = 'center', size = 'small') for i, equation in enumerate(equations)]
axes[1, 0].tick_params('both', left = False, bottom = False, labelleft = False, labelbottom = False)
__ = plt.setp(axes[1, 0], title = '2. Generate ODE System')

# Edit one of the rate laws
# G.edges[("S", "E")]["rate_law"] = r"\operatorname{z} \operatorname{S} \operatorname{I}"
G.edges[("S", "E")]["rate_law"] = r"z S I"

draw_model_diagram(G, ax = axes[0, 1])
__ = plt.setp(axes[0, 1], title = '3. Customize One Rate Law')

equations = extract_equations(G)
__ = [axes[1, 1].text(0.05, i / len(equations) + 0.5 / len(equations), s = "$" + equation + "$", va = 'center', size = 'small') for i, equation in enumerate(equations)]
axes[1, 1].tick_params('both', left = False, bottom = False, labelleft = False, labelbottom = False)
__ = plt.setp(axes[1, 1], title = '4. Regenerate ODE System')

# Convert LaTeX to MathML
equations_mathml = convert_latex2mathml(equations)
# __ = [print(eq + "\n<br>") for eq in equations_mathml]

# Convert to AMR
# data = {
#     "mathml": equations_mathml,
#     "model": "petrinet"
# }
# print(json.dumps(data, indent = 2))
model_amr = convert_mathml2amr(model_mathml = equations_mathml)
draw_petri(model_amr, legend = False, ax = axes[0, 2])
__ = plt.setp(axes[0, 2], title = '5. Convert to MathML and AMR')

axes[1, 2].remove()

fig.savefig('../figures/model_diagram_to_AMR.png', dpi = 150)

# %%









# %%
# Used \operatorname{}
# Does not work:
# {
#   "mathml": [
#     "<math><mfrac><mrow><mi>d</mi><mo>S</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>4</mn></msub><mo>-</mo><msub><mi>c</mi><mn>0</mn></msub><mo>S</mo></math>",
#     "<math><mfrac><mrow><mi>d</mi><mo>E</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>0</mn></msub><mo>S</mo><mo>-</mo><msub><mi>c</mi><mn>1</mn></msub><mo>E</mo></math>",
#     "<math><mfrac><mrow><mi>d</mi><mo>I</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>1</mn></msub><mo>E</mo><mo>-</mo><msub><mi>c</mi><mn>2</mn></msub><mo>I</mo><mo>-</mo><msub><mi>c</mi><mn>3</mn></msub><mo>I</mo></math>",
#     "<math><mfrac><mrow><mi>d</mi><mo>R</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>2</mn></msub><mo>I</mo></math>"
#   ],
#   "model": "petrinet"
# }

# Didn't use \operatorname{}
# Works:
# {
#   "mathml": [
#     "<math><mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>4</mn></msub><mo>-</mo><msub><mi>c</mi><mn>0</mn></msub><mi>S</mi></math>",
#     "<math><mfrac><mrow><mi>d</mi><mi>E</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>0</mn></msub><mi>S</mi><mo>-</mo><msub><mi>c</mi><mn>1</mn></msub><mi>E</mi></math>",
#     "<math><mfrac><mrow><mi>d</mi><mi>I</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>1</mn></msub><mi>E</mi><mo>-</mo><msub><mi>c</mi><mn>2</mn></msub><mi>I</mi><mo>-</mo><msub><mi>c</mi><mn>3</mn></msub><mi>I</mi></math>",
#     "<math><mfrac><mrow><mi>d</mi><mi>R</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><msub><mi>c</mi><mn>2</mn></msub><mi>I</mi></math>"
#   ],
#   "model": "petrinet"
# }

# %%



