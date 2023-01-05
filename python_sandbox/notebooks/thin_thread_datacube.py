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


scenario_descriptions = {}
for d in tqdm(dirs):
    p = os.path.join(root, d, 'description.json')
    with open(p, 'r') as f:
        scenario_descriptions[d] = json.load(f)['description']

# %%
# Get scenarios, attributes

scenarios = list(output.keys())
attributes = list(output[scenarios[0]].keys())
times = pd.to_datetime(output['1']['_time']['value'])


num_attributes = len(attributes)
num_scenarios = len(output)
num_replicates = 1
num_times = len(times)

# %%
# Load example data cube from Paul Cohen for reference
example_cube_Paul = xr.open_dataset('../data/Dedri Queries/cube.netcdf')

# Load example data cube from CIEMSS
example_cube_CIEMSS = xr.open_dataset('../data/Dedri Queries/ciemss_datacube.nc')

# %%
# Paul's example treats each attribute as its own `Data variable` with an attached 3D cube
# (dimensions = `scenarios`, `replicates`, `times`) and without any `Coordinates` values.
#
# CIEMSS' example is a 4D cube (dimensions = `experimental conditions`, `replicates`, `attributes`, `timesteps`) 
# with `Coordinates` values and one `Data variable` (= `__xarray_dataarray__`).

# %%
# Build xarray cube from `output` à la Paul
# Note: datetime type for `_time` attribute
 
output_cube_Paul = xr.Dataset(
    {
        attr: xr.DataArray(np.array([output[s][attr]['value'] if attr != '_time' else pd.to_datetime(output[s][attr]['value']) for s in scenarios])[:, np.newaxis, :], dims = ('scenarios', 'replicates', 'times'))
        for attr in attributes
    }
)

# %%
# Write to netCDF format
__ = output_cube_Paul.to_netcdf(path = f'{DATA_PATH}/output_Paul.nc', mode = 'w', format = 'NETCDF4')

# %%
# Repeat above à la CIEMSS

arr = np.array([[output[s][attr]['value'] for attr in attributes if attr != '_time'] for s in scenarios])
arr = np.reshape(arr, (num_scenarios, num_replicates, num_attributes - 1, num_times))

output_cube_CIEMSS = xr.Dataset(
    data_vars = {'__xarray_dataarray_variable__': (['scenarios', 'replicates', 'attributes', 'timesteps'], arr)},
    coords = {
        'scenarios': (['scenarios'], list(scenario_descriptions.values())),
        'replicates': (['replicates'], np.array([0])),
        'attributes': (['attributes'], [attr for attr in attributes if attr != '_time']),
        'timesteps': (['timesteps'], times),
    },
    attrs = {'description': 'Demo simulation outputs with the SIDARTHE model.'}
)

# %%
# Write to netCDF format
__ = output_cube_CIEMSS.to_netcdf(path = f'{DATA_PATH}/output_CIEMSS.nc', mode = 'w', format = 'NETCDF4')

# %%


