# %%[markdown]
# # Basic Run Example

# %%
from fair import FAIR
from fair.interface import fill, initialise
import matplotlib.pyplot as plt
import json

# %%
# 1. Create FaIR instance
f = FAIR()

# %%
# 2. Define time horizon of the simulation
# From the year 2000 to 2050 in steps of 1 year
f.define_time(2000, 2050, 1)

print(f.timebounds)

print(f.timepoints)

# %%
# 3. Define label of scenarios
# One scenario named "abrupt" where emissions/concentrations change instantly
f.define_scenarios(["abrupt"])
f.scenarios

# %%
# 4. Define config labels
# 3 config sets corresponding to high, medium, low climate sensitivity
f.define_configs(["high", "central", "low"])
f.configs

# %%
# 5. Define species
# "species" = anthropogenic or natural forcers in the scenario

# Define label of 8 species
# * CO2 emissions from fossil and industry
# * CO2 emissions from agriculture, forestry, and other land uses (AFOLU)
# * Sulfur emissions
# * CH4 concentration
# * N2O concentration
# * CO2 (calculated concentration)
# * aerosol radiation (ERFari)
# * aerosol cloud interactions (ERFaci)
species = ['CO2 fossil emissions', 'CO2 AFOLU emissions', 'Sulfur', 'CH4', 'N2O', 'CO2', 'ERFari', 'ERFaci']

# Define species behaviour
properties = {
    'CO2 fossil emissions': {
        'type': 'co2 ffi',
        'input_mode': 'emissions',
        'greenhouse_gas': False,  # it doesn't behave as a GHG itself in the model, but as a precursor
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    },
    'CO2 AFOLU emissions': {
        'type': 'co2 afolu',
        'input_mode': 'emissions',
        'greenhouse_gas': False,  # it doesn't behave as a GHG itself in the model, but as a precursor
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    },
    'CO2': {
        'type': 'co2',
        'input_mode': 'calculated',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    },
    'CH4': {
        'type': 'ch4',
        'input_mode': 'concentration',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': True, # we treat methane as a reactive gas
    },
    'N2O': {
        'type': 'n2o',
        'input_mode': 'concentration',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': True, # we treat nitrous oxide as a reactive gas
    },
    'Sulfur': {
        'type': 'sulfur',
        'input_mode': 'emissions',
        'greenhouse_gas': False,
        'aerosol_chemistry_from_emissions': True,
        'aerosol_chemistry_from_concentration': False,
    },
    'ERFari': {
        'type': 'ari',
        'input_mode': 'calculated',
        'greenhouse_gas': False,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    },
    'ERFaci': {
        'type': 'aci',
        'input_mode': 'calculated',
        'greenhouse_gas': False,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False,
    }
}

f.define_species(species, properties)

# %%
# 6. Modify run options

# Available options
#  |  Parameters
#  |  ----------
#  |  n_gasboxes : int
#  |      the number of atmospheric greenhouse gas boxes to run the model with
#  |  n_layers : int
#  |      the number of ocean layers in the energy balance or impulse
#  |      response model to run with
#  |  iirf_max : float
#  |      limit for time-integral of greenhouse gas impulse response function.
#  |  br_cl_ods_potential : float
#  |      factor describing the ratio of efficiency that each bromine atom
#  |      has as an ozone depleting substance relative to each chlorine atom.
#  |  ghg_method : str
#  |      method to use for calculating greenhouse gas forcing from CO\ :sub:`2`,
#  |      CH\ :sub:`4` and N\ :sub:`2`\ O. Valid options are {"leach2021",
#  |      "meinshausen2020", "etminan2016", "myhre1998"}
#  |  ch4_method : str
#  |      method to use for calculating methane lifetime change. Valid options are
#  |      {"leach2021", "thornhill2021"}.
#  |  temperature_prescribed : bool
#  |      Run FaIR with temperatures prescribed.
help(f)

print(f.ghg_method)
# "meinshausen2020"

f.aci_method = "myhre1998"
print(f.aci_method)

# %%
# 7. Create input and output data
# Allocate data arrays for `emissions` and `temperature`

f.allocate()

f.emissions

f.temperature

# %%
# 8. Fill in the data

# %%
# 8a. Fill in the emissions data
# f.emissions.loc[(dict(specie="CO2 fossil emissions", scenario="abrupt"))] = 38
fill(f.emissions, 38, scenario='abrupt', specie='CO2 fossil emissions')
fill(f.emissions, 3, scenario='abrupt', specie='CO2 AFOLU emissions')
fill(f.emissions, 100, scenario='abrupt', specie='Sulfur')
fill(f.concentration, 1800, scenario='abrupt', specie='CH4')
fill(f.concentration, 325, scenario='abrupt', specie='N2O')

# Define initial condition at first timestep
initialise(f.concentration, 278.3, specie='CO2')
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
# 8b. Fill in the `climate_configs` (model response to a forcing)
# this represents the behaviour of a three-layer energy balance model

fill(f.climate_configs["ocean_heat_transfer"], [0.6, 1.3, 1.0], config='high')
fill(f.climate_configs["ocean_heat_capacity"], [5, 15, 80], config='high')
fill(f.climate_configs["deep_ocean_efficacy"], 1.29, config='high')

fill(f.climate_configs["ocean_heat_transfer"], [1.1, 1.6, 0.9], config='central')
fill(f.climate_configs["ocean_heat_capacity"], [8, 14, 100], config='central')
fill(f.climate_configs["deep_ocean_efficacy"], 1.1, config='central')

fill(f.climate_configs["ocean_heat_transfer"], [1.7, 2.0, 1.1], config='low')
fill(f.climate_configs["ocean_heat_capacity"], [6, 11, 75], config='low')
fill(f.climate_configs["deep_ocean_efficacy"], 0.8, config='low')


# %%
# 8c. Fill in `species_configs` (behaviour and properties of species)

f.species_configs
# FAIR.fill_species_configs() <= load defaults


# Greenhouse gas state dependence
fill(f.species_configs["partition_fraction"], [0.2173, 0.2240, 0.2824, 0.2763], specie="CO2")

non_co2_ghgs = ["CH4", "N2O"]
for gas in non_co2_ghgs:
    fill(f.species_configs["partition_fraction"], [1, 0, 0, 0], specie=gas)

fill(f.species_configs["unperturbed_lifetime"], [1e9, 394.4, 36.54, 4.304], specie="CO2")
fill(f.species_configs["unperturbed_lifetime"], 8.25, specie="CH4")
fill(f.species_configs["unperturbed_lifetime"], 109, specie="N2O")

fill(f.species_configs["baseline_concentration"], 278.3, specie="CO2")
fill(f.species_configs["baseline_concentration"], 729, specie="CH4")
fill(f.species_configs["baseline_concentration"], 270.3, specie="N2O")

fill(f.species_configs["forcing_reference_concentration"], 278.3, specie="CO2")
fill(f.species_configs["forcing_reference_concentration"], 729, specie="CH4")
fill(f.species_configs["forcing_reference_concentration"], 270.3, specie="N2O")

fill(f.species_configs["molecular_weight"], 44.009, specie="CO2")
fill(f.species_configs["molecular_weight"], 16.043, specie="CH4")
fill(f.species_configs["molecular_weight"], 44.013, specie="N2O")

fill(f.species_configs["greenhouse_gas_radiative_efficiency"], 1.3344985680386619e-05, specie='CO2')
fill(f.species_configs["greenhouse_gas_radiative_efficiency"], 0.00038864402860869495, specie='CH4')
fill(f.species_configs["greenhouse_gas_radiative_efficiency"], 0.00319550741640458, specie='N2O')


# Compute the baseline time-integrated airborn fraction `iirf_0`
# from lifetime, molecular weight and partition fraction
f.calculate_iirf0()
f.calculate_g()
f.calculate_concentration_per_emission()

# Manual override possible
# fill(f.species_configs["iirf_0"], 29, specie='CO2')

# Define sensitivities of airborn fraction for each GHG
fill(f.species_configs["iirf_airborne"], [0.000819*2, 0.000819, 0], specie='CO2')
fill(f.species_configs["iirf_uptake"], [0.00846*2, 0.00846, 0], specie='CO2')
fill(f.species_configs["iirf_temperature"], [8, 4, 0], specie='CO2')

fill(f.species_configs['iirf_airborne'], 0.00032, specie='CH4')
fill(f.species_configs['iirf_airborne'], -0.0065, specie='N2O')

fill(f.species_configs['iirf_uptake'], 0, specie='N2O')
fill(f.species_configs['iirf_uptake'], 0, specie='CH4')

fill(f.species_configs['iirf_temperature'], -0.3, specie='CH4')
fill(f.species_configs['iirf_temperature'], 0, specie='N2O')


# Aerosol emissions or concentrations to forcing
fill(f.species_configs["erfari_radiative_efficiency"], -0.0036167830509091486, specie='Sulfur') # W m-2 MtSO2-1 yr
fill(f.species_configs["erfari_radiative_efficiency"], -0.002653/1023.2219696044921, specie='CH4') # W m-2 ppb-1
fill(f.species_configs["erfari_radiative_efficiency"], -0.00209/53.96694437662762, specie='N2O') # W m-2 ppb-1

fill(f.species_configs["aci_scale"], -2.09841432)
fill(f.species_configs["aci_shape"], 1/260.34644166, specie='Sulfur')

# %%
# Save inputs
# list: https://docs.fairmodel.net/en/latest/intro.html#state-variables

# simulation parameters
with open("./inputs/simulation_parameters.json", "w") as i:
    json.dump({
        "time": {"start": 2000, "end": 2050, "step": 1},
        "scenarios": ['abrupt'],
        "configs": ['high', 'central', 'low'],
        "species": {"labels": species, "properties": properties}
    }, i, indent = 2)


# state variable initial conditions
f.emissions.to_netcdf("./inputs/emissions.nc")
f.concentration.to_netcdf("./inputs/concentration.nc")
f.forcing.to_netcdf("./inputs/forcing.nc")
f.temperature.to_netcdf("./inputs/temperature.nc")
f.airborne_emissions.to_netcdf("./inputs/airborne_emisssions.nc")
f.airborne_fraction.to_netcdf("./inputs/airborne_fraction.nc")
f.cumulative_emissions.to_netcdf("./inputs/cumulative_emissions.nc")

# Not required
# f.ocean_heat_content_change.to_netcdf("./outputs/ocean_heat_content_change.nc")
# f.stochastic_forcing.to_netcdf("./outputs/stochastic_forcing.nc")
# f.toa_imbalance.to_netcdf("./outputs/toa_imbalance.nc")

# Configs
f.climate_configs.to_netcdf("./inputs/climate_configs.nc")
f.species_configs.to_netcdf("./inputs/species_configs.nc")

# %%
# 9. Run
f.run()

# %%
# 10. Plot results

fig, axes = plt.subplots(2, 2, figsize = (10, 10))
fig.suptitle("Abrupt Scenario")
for (ax, y, yl) in zip(fig.axes, [f.temperature.loc[dict(scenario='abrupt', layer=0)], f.forcing_sum.loc[dict(scenario='abrupt')], f.concentration.loc[dict(scenario='abrupt', specie='CO2')], f.forcing.loc[dict(scenario='abrupt', specie='ERFaci')]], ['Temperature anomaly (K)', 'Total ERF (W m$^{-2}$)', 'CO2 (ppm)', 'ERF from aerosol-cloud interactions (W m$^{-2}$)']):
    __ = ax.plot(f.timebounds, y, label = f.configs)
    __ = plt.setp(ax, ylabel = yl)
    __ = ax.legend()

fig.savefig("basic_run_example_output.png", dpi = 150)

# %%
f.species_configs['g0'].loc[dict(specie='CO2')]

# %%
f.forcing[-1, :, 1, :]

# %%
# Save outputs

f.emissions.to_netcdf("./outputs/emissions.nc")
f.concentration.to_netcdf("./outputs/concentration.nc")
f.forcing.to_netcdf("./outputs/forcing.nc")
f.temperature.to_netcdf("./outputs/temperature.nc")
f.airborne_emissions.to_netcdf("./outputs/airborne_emisssions.nc")
f.airborne_fraction.to_netcdf("./outputs/airborne_fraction.nc")
f.cumulative_emissions.to_netcdf("./outputs/cumulative_emissions.nc")

f.ocean_heat_content_change.to_netcdf("./outputs/ocean_heat_content_change.nc")
f.stochastic_forcing.to_netcdf("./outputs/stochastic_forcing.nc")
f.toa_imbalance.to_netcdf("./outputs/toa_imbalance.nc")

# %%
