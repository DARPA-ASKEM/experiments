# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Convert Paul Cohen's `xarray` netCDF data-cube example to 
# input format of the 'responsive cell matrix' (RCM) renderer

# %%
# Input arguments:
# * the data cube
# * dimension-axis mapping (x, y, z)
# * transform options (centroid, value_counts, etc.)

# %%
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

# %%
# Load data cubes

# Example cubes
# example_cube_Paul = xr.open_dataset('../data/Dedri Queries/cube.netcdf')
# example_cube_CIEMSS = xr.open_dataset('../data/Dedri Queries/ciemss_datacube.nc')

# Demo cube à la CIEMSS
cube = xr.open_dataset('../../thin-thread-examples/demo/BIOMD0000000955/runs/output_CIEMSS.nc')

# %%
# Data array
data = cube['__xarray_dataarray_variable__']

# %%
# Specify what to do with each cube dimension
map_cube_matrix = {
    'scenarios': 'rows',
    'replicates': 'select 0',
    'attributes': 'vars',
    'timesteps': 'cols'
}

# %%
# Define what the rows, cols, and vars of the matrix are
map_matrix_cube = {v: k for k, v in map_cube_matrix.items() if v in ('rows', 'cols', 'vars')}

# %%
# Initialize `matrix` with given row, col definition

matrix = {
    str(i): {
        'description': s.data.item(), 
        'initials': {},
        'parameters': {},
        'output': {
            str(k): {
                'name': str(v.data.item()),
                'identifiers': {},
                'value': []
            }
            for k, v in enumerate(cube[map_matrix_cube['vars']])
        }
    }
    for i, s in enumerate(cube[map_matrix_cube['rows']])
}


# %%
# Map dimension associated with `rows` and `vars` to cube axis index
map_dim_ax = {d: i for i, d in enumerate(cube.dims)}
rows_axis = map_dim_ax[map_matrix_cube['rows']]
vars_axis = map_dim_ax[map_matrix_cube['vars']]

# %%
# Populate matrix by slicing cube
for i, s in enumerate(cube[map_matrix_cube['rows']]):
    for k, v in enumerate(cube[map_matrix_cube['vars']]):

        # Define slice of cube
        slc = [slice(None)] * len(cube['__xarray_dataarray_variable__'].shape)
        
        # Select i-th row and k-th variable
        slc[rows_axis] = slice(i, i + 1)
        slc[vars_axis] = slice(k, k + 1)

        # Check for dim-wise `select` operations
        for dim, op in map_cube_matrix.items():

            # Select n-th element in `op` dimension
            if op.split(' ')[0] == 'select':
                op_axis = map_dim_ax[dim]
                n = int(op.split(' ')[1])
                slc[op_axis] = slice(n, n + 1)

        # Check for dim-wise `centroid` operation
        # Only one such operation allowed
        for dim, op in map_cube_matrix.items():

            # Find centroid 
            if op == 'centroid':
                pass
                
                # cube_slice = cube['__xarray_dataarray_variable__'][tuple(slc)].squeeze()
            
                # Compute centroid
                # c0 = cube_slice.median(dim = dim)
                
                # Build kd-tree
                # kdtree = sp.spatial.KDTree(data)
                
                # Find point nearest to centroid
                # c1 = kdtree.query(c0, 1, p = 2)
                # matrix[str(i)]['output'][str(k)]['value'] = list(c1)


        # Extract slice as variable values
        cube_slice = cube['__xarray_dataarray_variable__'][tuple(slc)].squeeze()
        matrix[str(i)]['output'][str(k)]['value'] = list(cube_slice.data)

# %%
# Save as JSON
with open('../../thin-thread-examples/demo/BIOMD0000000955/runs/output_CIEMSS_matrix.json', 'w') as f:
    f.write(json.dumps(matrix, indent = 4))

# %%
