# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# Create a "Responsive Matrix Cell" input file from the PyCIEMSS integration demo results

# %%
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# Load PyCIEMSS integration demo results
# (from `load_and_sample_petri_model`)

RESULT_PATH = '../../thin-thread-examples/integration/pyciemss/demo_results'

pyciemss_results = {
    'parameters': {},
    'states': {}
}

__, __, filenames = next(os.walk(RESULT_PATH))
for filename in filenames:
    path = os.path.join(RESULT_PATH, filename)
    name, k = filename.split('.')[0].split('_')
    if k == 'param':
        pyciemss_results['parameters'][name] = np.genfromtxt(path)
    elif k == 'sol':
        pyciemss_results['states'][name] = np.genfromtxt(path)
    else:
        pass

# %%
# Build RCM data structure

rcm_input = {
    '0': {
        'description': 'PyCIEMSS integration demo', 
        'initials': {
            str(i): {
                'name': k,
                'identifiers': {},
                'value': list(v[:, 0])
            }
            for i, (k, v) in enumerate(pyciemss_results['states'].items())
        },
        'parameters': {
            str(i): {
                'name': k,
                'identifiers': {},
                'value': list(v)
            }
            for i, (k, v) in enumerate(pyciemss_results['parameters'].items())
        },
        'output': {
            str(i): {
                'name': k,
                'identifiers': {},
                'value': v.tolist()
            }
            for i, (k, v) in enumerate(pyciemss_results['states'].items())
        }
    }
}

# %%
# Write RCM JSON

with open(os.path.join(RESULT_PATH, 'rcm_input.json'), 'w') as f:
    f.write(json.dumps(rcm_input, indent = 4))

# %%
