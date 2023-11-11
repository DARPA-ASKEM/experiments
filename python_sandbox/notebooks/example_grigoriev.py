# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # Evaluate Grigoriev Ice Cap Example

# %%
import PIL
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
PATH = "../../notebooks/Nelson/decapodes/dome_model/grigoriev/inputs/LanderVT-Icethickness-Grigoriev-ice-cap-c32c9f7"

# %%
# ## 1. Transform Dataset
# 
# 

# %%
# Load ice thickness as TIF image
with PIL.Image.open(PATH + "/Icethickness_Grigoriev_ice_cap_2021.tif") as img:
    h_init_tif = np.array(img)
    print(f"Width x Height = {h_init_tif.shape[0]} px x {h_init_tif.shape[1]} px")

# Load measurement tracks
measure_df = pd.read_excel(PATH + "/" + "measurements_2021.xlsx")
flow_df = pd.read_excel(PATH + "/" + "extendedflowlines_2021.xlsx")

# Load ShapeFile of the outline
outline = gpd.read_file(PATH + "/" + "Outline_2021.shp")

# %%
# Bounds of the outline = bounds of the TIF image
MIN_X, MIN_Y, MAX_X, MAX_Y = outline["geometry"].values[0].bounds

print(f"X = {MIN_X} to {MAX_X}\nY = {MIN_Y} to {MAX_Y}")

# %%
# Zero -INF values
h_init_tif[h_init_tif < 0] = None

# %%
fig, ax = plt.subplots(1, 2, figsize = (10, 8))
fig.suptitle("Thickness of Grigoriev Ice Cap")

m = ax[0].imshow(h_init_tif, interpolation = "none")
__ = plt.setp(ax[0], xlabel = "X", ylabel = "Y")
__ = fig.colorbar(m, ax = ax[0], location = "top")

m = ax[1].imshow(h_init_tif, interpolation = "none", extent = (MIN_X, MAX_X, MIN_Y, MAX_Y))

l = np.array([list(p) for p in outline["geometry"].values[0].exterior.coords])
__ = ax[1].plot(l[:, 0], l[:, 1], marker = None, color = 'r', label = "Outline")
__ = ax[1].scatter(measure_df["X"], measure_df["Y"], marker = '.', alpha = 0.5, label = "Measurements")
__ = ax[1].scatter(flow_df["X"], flow_df["Y"], marker = ".", alpha = 0.5, label = "Flowlines")
__ = ax[1].legend()
__ = fig.colorbar(m, ax = ax[1], location = "top")
__ = plt.setp(ax[1], xlabel = "X", ylabel = "Y")

pos0 = plt.getp(ax[0], "position").bounds
pos1 = plt.getp(ax[1], "position").bounds
__ = plt.setp(ax[1], position = [pos1[0], pos0[1], pos0[2], pos0[3]])

fig.savefig("../../notebooks/Nelson/decapodes/dome_model/grigoriev/input_dataset_plot.png", dpi = 150)

# %%[markdown]
# Note: 
# * X axis = East-West and Y axis = North-South
# * https://maps.app.goo.gl/pcyHZtAhr4UnNUS9A
# * https://tc.copernicus.org/articles/17/4315/2023/tc-17-4315-2023.pdf

# %%
