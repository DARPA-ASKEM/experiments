# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%

# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage as ski
# from typing import NoReturn, Optional, Any

# %%
def map_value_colour(value: float, min_value: float, max_value: float, log_base: int = 10) -> str:

    # Check constraints
    if (value < min_value) | (value > max_value):
        raise ValueError('`value` must between `min_value` and `max_value`.')
    if (min_value <= 0) | (max_value <= 0):
        raise ValueError('`value`, `min_value`, and `max_value` must be greater than zero.')
    if (isinstance(log_base, int) == False) | (log_base <= 0):
        raise ValueError('`log_base` must be an integer that is greater than zero.')
    
    # Compute log range
    log_min_value = int(np.floor(np.log(min_value) / np.log(log_base)))
    log_max_value = int(np.ceil(np.log(max_value) / np.log(log_base)))
    num_log_orders = int(log_max_value - log_min_value)

    # Minimum log range = 1
    if num_log_orders < 1:
        num_log_orders = 1

    # 10 orders -> Tableau 10
    if num_log_orders < 10:

        palette = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

    # 40 orders -> Tableau 40
    elif num_log_orders < 40:

        palette = ['#03579b','#0488d1','#03a9f4','#4fc3f7','#b3e5fc','#253137','#455a64','#607d8b','#90a4ae','#cfd8dc','#19237e','#303f9f','#3f51b5','#7986cb','#c5cae9','#4a198c','#7b21a2','#9c27b0','#ba68c8','#e1bee7','#88144f','#c21f5b','#e92663','#f06292','#f8bbd0','#bf360c','#e64a18','#ff5722','#ff8a65','#ffccbc','#f67f17','#fbc02c','#ffec3a','#fff177','#fdf9c3','#33691d','#689f38','#8bc34a','#aed581','#ddedc8']
    
    # Truncate colour palette
    palette_ = palette[:num_log_orders]

    # Convert to RGB
    palette_RGB = np.array([[int(c.lstrip('#')[i:(i + 2)], 16) / 255.0 for i in (0, 2, 4)] for c in palette_])

    # Convert to LAB
    palette_LAB = np.array([ski.color.rgb2lab(c) for c in palette_RGB])

    # Index of `value` in palette array
    log_value_index = int(np.log(value) / np.log(log_base)) - log_min_value

    # Pick the two palette colours nearest to the given value
    if log_value_index < num_log_orders:
        i = log_value_index
        j = log_value_index + 1
    else:
        i = log_value_index
        j = -1

    # Interpolate by lightness
    value_colour = palette_LAB[i, :]
    m = np.log(value) / np.log(log_base) - int(np.log(value) / np.log(log_base))
    value_colour[2] = (palette_LAB[j, 2] - palette_LAB[i, 2]) / (1) * m + palette_LAB[i, 2]

    # Convert back to RGB
    return ski.color.lab2rgb(value_colour)

# %%
t = np.linspace(0, 2 * np.pi, 1000)
data = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

fig, ax = plt.subplots(1, 1)
h = ax.imshow(data)
__ = fig.colorbar(h)


# %%
# Current color function

# const fillColorFn = (datum: CellData, parametersMin: any, parametersMax: any) => {
#     const colorExtremePos = '#4d9221';
#     const colorExtremeNeg = '#c51b7d';
#     const colorMid = '#f7f7f7';
#     const datumBase: any = data[baseRow][datum.col];
#     if (datum.row === baseRow) {
#         const v =
#             (datum.Infected - parametersMin.Infected) /
#             (parametersMax.Infected - parametersMin.Infected);
#         return mix('#F7F7F7', '#252525', v, 'lab');
#     }
#     // return midpoint color if the range of Infected is 0 to avoid divide-by-0 error
#     if (!divergingMaxMin) {
#         return colorMid;
#     }
#     const diff = datum.Infected - datumBase.Infected;
#     // map ratio value to a hex colour
#     if (diff >= 0) {
#         return mix(colorMid, colorExtremePos, diff / divergingMaxMin, 'lab');
#     }
#     return mix(colorMid, colorExtremeNeg, -diff / divergingMaxMin, 'lab');
# };

# %%
