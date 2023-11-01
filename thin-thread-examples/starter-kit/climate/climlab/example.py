# %%[markdown]
# Tutorial Example - Preconfigured Energy Balance Model
# 
# Reference: https://climlab.readthedocs.io/en/latest/courseware/Preconfigured_EBM.html

# %%
import numpy as np
import matplotlib.pyplot as plt
import climlab
from climlab import constants as const
import json

# %%
# 1. Create model

ebm_model = climlab.EBM(name = "preconfigured energy balance model")

# Default model parameters
# num_lat=90, S0=const.S0, A=210., B=2., D=0.55, water_depth=10., Tf=-10, a0=0.3, a2=0.078, ai=0.62, timestep=const.seconds_per_year/90., T0=12., T2=-40
print(ebm_model.param)

# Save inputs
with open("./inputs/simulation_parameters.json", "w") as f:
    json.dump(ebm_model.param, f, indent = 2)

# %%
# Only 1 state variable = `Ts` (temperature)
# "subprocesses"
print(ebm_model)

# time coordinate
print(ebm_model.time)

# %%
# 2. Run simulation by integrating over 50 days + 1 year
ebm_model.integrate_days(50.0)
ebm_model.integrate_years(1.0)

# %%
# 3. Inspect model variables

print(list(ebm_model.diagnostics.keys()))
# ['OLR', 'insolation', 'coszen', 'icelat', 'ice_area', 'albedo', 'ASR', 'diffusive_flux', 'advective_flux', 'total_flux', 'flux_convergence', 'heat_transport', 'heat_transport_convergence', 'net_radiation']

# %%
# Save output as NetCDF
ebm_model.to_xarray(diagnostics = True).to_netcdf("./outputs/outputs.nc")

# What is `timeave`?
# climlab.to_xarray(ebm_model.timeave).to_netcdf("./outputs/outputs_timeave.nc")

# %%
# 4. Creating plot figure
fig = plt.figure(figsize=(15,10))

# Temperature plot
ax1 = fig.add_subplot(221)
ax1.plot(ebm_model.lat,ebm_model.Ts)

ax1.set_xticks([-90,-60,-30,0,30,60,90])
ax1.set_xlim([-90,90])
ax1.set_title('Surface Temperature', fontsize=14)
ax1.set_ylabel('(degC)', fontsize=12)
ax1.grid()

# Albedo plot
ax2 = fig.add_subplot(223, sharex = ax1)
ax2.plot(ebm_model.lat,ebm_model.albedo)

ax2.set_title('Albedo', fontsize=14)
ax2.set_xlabel('latitude', fontsize=10)
ax2.set_ylim([0,1])
ax2.grid()

# Net Radiation plot
ax3 = fig.add_subplot(222, sharex = ax1)
ax3.plot(ebm_model.lat, ebm_model.OLR, label='OLR', color='cyan')
ax3.plot(ebm_model.lat, ebm_model.ASR, label='ASR', color='magenta')
ax3.plot(ebm_model.lat, ebm_model.ASR-ebm_model.OLR, label='net radiation', color='red')

ax3.set_title('Net Radiation', fontsize=14)
ax3.set_ylabel('(W/m$^2$)', fontsize=12)
ax3.legend(loc='best')
ax3.grid()

# Energy Balance plot
net_rad = np.squeeze(ebm_model.net_radiation)
transport = np.squeeze(ebm_model.heat_transport_convergence)

ax4 = fig.add_subplot(224, sharex = ax1)
ax4.plot(ebm_model.lat, net_rad, label='net radiation', color='red')
ax4.plot(ebm_model.lat, transport, label='heat transport', color='blue')
ax4.plot(ebm_model.lat, net_rad+transport, label='balance', color='black')

ax4.set_title('Energy', fontsize=14)
ax4.set_xlabel('latitude', fontsize=10)
ax4.set_ylabel('(W/m$^2$)', fontsize=12)
ax4.legend(loc='best')
ax4.grid()

fig.savefig("example.png", dpi = 150)

# %%
