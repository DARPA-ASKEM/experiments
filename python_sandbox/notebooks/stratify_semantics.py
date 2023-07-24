# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
import json

# %%
models = {}
model_names = ('sir_typed_aug', 'flux_typed_aug', 'sir_flux_span')

for k in model_names:
    with open(f'../data/petrinet_examples/{k}.json', 'r') as f:
        models[k] = json.load(f)

# %%
# Map between stratified-model nodes and their parent base/factor-model nodes
maps = {node['id']: [] for node_type in ('states', 'transitions') for node in models[model_names[-1]]['model'][node_type]}
for span in models[model_names[-1]]['semantics']['span']:
    for node in span['map']:
        maps[node[0]].append(node[1])

# List of stratified-model transition nodes
transitions = models[model_names[-1]]['model']['transitions']


for transition in transitions:

    # Default expression
    expression = '*'.join(transition['input'])

    # parent transitions
    expression_param = []
    for i, parent_transition in enumerate(maps[transition['id']]):

        rates = models[model_names[-1]]['semantics']['span'][i]['system']['semantics']['ode']['rates']
        m = {r['target']: r['expression'] for r in rates}

        expression_param.append(m[parent_transition])

