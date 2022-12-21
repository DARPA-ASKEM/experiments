# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Convert the simulation runs in `thin-thread examples/demo/BIOMD0000000955` 
# into Paul Cohen's `xarray` netCDF data-cube format

# %%
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

# %%
DATA_PATH = '../../thin-thread-examples/demo/BIOMD0000000955/runs'

# %%
# Load data from SIDARTHE demo runs
root, dirs, files = next(os.walk(DATA_PATH))
output = {}
for d in tqdm(dirs):
    p = os.path.join(root, d, 'output.json')
    with open(p, 'r') as f:
        output[d] = json.load(f)

# %%
# Get scenarios, attributes

scenarios = list(output.keys())
attributes = list(output[scenarios[0]].keys())

num_attributes = len(attributes)
num_scenarios = len(output)
num_replicates = 1
num_times = len(output[scenarios[0]]['_time']['value'])

# %%
# Load example "data cube" from Paul Cohen for reference
example_cube = xr.open_dataset('../data/Dedri Queries/cube.netcdf')

# %%
# Build xarray cube from `output`
# Note: datetime type for `_time` attribute

output_cube = xr.Dataset(
    {
        attr: xr.DataArray(np.array([output[s][attr]['value'] if attr != '_time' else pd.to_datetime(output[s][attr]['value']) for s in scenarios])[:, np.newaxis, :], dims = ('scenarios', 'replicates', 'times'))
        for attr in attributes
    }
)

# %%
# Write `output_cube` in netCDF format
__ = output_cube.to_netcdf(path = f'{DATA_PATH}/output.nc', mode = 'w', format = 'NETCDF4')

# %%