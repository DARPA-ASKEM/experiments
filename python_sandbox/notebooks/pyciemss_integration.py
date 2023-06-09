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
import pandas as pd
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
# Alternate: flat dataframe CSV

# Time points and sample points
num_samples, num_timepoints = next(iter(pyciemss_results['states'].values())).shape
d = {
    'timepoint_id': np.tile(np.array(range(num_timepoints)), num_samples),
    'sample_id': np.repeat(np.array(range(num_samples)), num_timepoints)
}

# Parameters
d = {**d, **{f'{k}_param': np.repeat(v, num_timepoints) for k, v in pyciemss_results['parameters'].items()}}

# Solution (state variables)
d = {**d, **{f'{k}_sol': np.squeeze(v.reshape((num_timepoints * num_samples, 1))) for k, v in pyciemss_results['states'].items()}}

df = pd.DataFrame(d)

# Write to CSV
df.to_csv(os.path.join(RESULT_PATH, 'pyciemss_results.csv'), index = False)

# %%