# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # Time-series as Funman Input
#
# 1. Start with a time-series dataset for a given state variable
# 2. Convert it to a sequence of constraints compatible with Funman
# 3. Run Funman
# 4. Inspect results

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

# %%
# Get test model with PyCIEMSS simulate results
model_url = "https://raw.githubusercontent.com/liunelson/pyciemss/nl-test/notebooks/data/test_simulate_model.json"
simulate_results_url = "https://raw.githubusercontent.com/liunelson/pyciemss/nl-test/notebooks/data/test_simulate_interface.csv"
PATH = "../data/example4"

# %%
r = requests.get(model_url)
if r.ok:
    model = r.json()
    with open(os.path.join(PATH, "model.json"), "w") as fp:
        json.dump(model, fp, indent = 4)

model

# %%
r = requests.get(simulate_results_url)
if r.ok:
    with open(os.path.join(PATH, "simulate_results.csv"), "wb") as fp:
        fp.write(r.content)

simulate_results = pd.read_csv(os.path.join(PATH, "simulate_results.csv"))
simulate_results.head()

# %%
# Construct Funman request
def generate_request(model: dict, timepoints: list, constraints: list[dict] = [], use_compartmental_constraints: bool = True, normalization_constant: float = 1, tolerance: float = 0.1, parameters_of_interest: list[str] = []) -> dict:

    # Define constraints
    constraints = constraints

    # Note: Linear constraint example
    # 0.0 <= 1.0 * beta + 1.0 * gamma <= 1.0
    # linear_constraint = {
    #     "name": "example",
    #     "additive_bounds": {
    #         "lb": 0.0,
    #         "ub": 1.0
    #     },
    #     "variables": ["beta", "gamma"],
    #     "weights": [1.0, 1.0]
    # }


    # Define parameter limits
    parameters = []
    for p in model["semantics"]["ode"]["parameters"]:

        if p["id"] in parameters_of_interest:
            label = "all"
        else:
            label = "any"


        if "distribution" in p.keys():
            parameters.append(
                {
                    "name": p["id"],
                    "interval": {
                        "lb": p["distribution"]["parameters"]["minimum"],
                        "ub": p["distribution"]["parameters"]["maximum"]
                    },
                    "label": label
                }
            )

        else:
            parameters.append(
                {
                    "name": p["id"],
                    "interval": {
                        "lb": p["value"],
                        "ub": p["value"]
                    },
                    "label": label
                }
            )

    # Generate request object
    request = {
        "model": model,
        "request": {
            "constraints": constraints,
            "parameters": parameters,
            "structure_parameters": [{
                "name": "schedules",
                "schedules": [{
                    "timepoints": timepoints
                }]
            }],
            "config": {
                "use_compartmental_constraints": use_compartmental_constraints,
                "normalization_constant": normalization_constant,
                "tolerance": tolerance
            }
        }
    }

    return request

# %%
# Generate Funman constraint constraints from a time-series dataframe
def generate_constraints_from_timeseries(model: dict, df: pd.DataFrame, mapping_df_model: dict = None, num_constraints: int = 2, plot: bool = False) -> dict:

    # Compute number of tolerance intervals
    try:
        assert num_constraints > 0
    except AssertionError:
        print("`num_constraints` must be greater than 0.")

    # Mapping between model variables and dataframe variables
    # If None, assume df column
    if isinstance(mapping_df_model, type(None)):
        mapping_df_model = {c: "_".join(c.split("_")[:-1]) for c in df.columns if (c.split("_")[-1] == "state") & ("observable" not in c.split("_"))}
        mapping_df_model["timepoint_unknown"] = "timepoints"
    
    # Check mapping
    model_states = [s["id"] for s in model["model"]["states"]]
    for k, v in mapping_df_model.items():
        if v != "timepoints":
            try:
                assert k in df.columns
            except:
                print(f"Mapping key `{k}` is not a `df` column header")
            
            try:
                assert v in model_states
            except:
                print(f"Mapping value `{v}` is not a state variable in `model`")


    # Timepoints and upper/lower limits of the given time-series dataset
    # Handle if `sample_id` is present in dataframe (from PyCIEMSS Simulate)
    k_timepoints = [k for k, v in mapping_df_model.items() if v == "timepoints"][0]
    timepoints = df[k_timepoints].unique()

    # Range of timepoints
    timepoint_min = min(timepoints)
    timepoint_max = max(timepoints)

    # Define constraint constraints
    constraints = []
    for k_df, k_model in mapping_df_model.items():

        if k_model == "timepoints":
            continue

        timepoints_constraint = np.linspace(timepoint_min, timepoint_max, int(num_constraints) + 1)

        for t in range(num_constraints):

            t1 = timepoints_constraint[t]
            t2 = timepoints_constraint[t + 1]

            constraints.append(
                {
                    "name": f"{k_model}_constraint{t}",
                    "variable": k_model,
                    "interval": {
                        "lb": min(df[(df[k_timepoints] >= t1) & (df[k_timepoints] <= t2)][k_df]),
                        "ub": max(df[(df[k_timepoints] >= t1) & (df[k_timepoints] <= t2)][k_df])
                    },
                    "timepoints": {
                        "lb": t1,
                        "ub": t2,
                        "closed_upper_bound": True
                    }
                }
            )

    if plot == True:

        if k_timepoints in mapping_df_model:
            mapping_df_model.pop(k_timepoints)

        num_states = len(mapping_df_model)
        colors = mpl.colormaps["tab10"](range(10))

        fig, ax = plt.subplots(num_states, 1, figsize = (12, 4 * num_states))

        for ax, color, (k_df, k_model) in zip(fig.axes, colors, mapping_df_model.items()):

            if k_model == "timepoints":
                pass

            # Plot time-series dataset
            x = timepoints
            y = df.groupby(["timepoint_id"]).mean()[k_df]
            __ = ax.plot(x, y, linewidth = 2, color = color)
            num_samples = len(df["sample_id"].unique())
            for n in range(num_samples):
                y = df[df["sample_id"] == n][k_df]
                __ = ax.plot(x, y, label = n, linewidth = 0.5, alpha = 0.5, color = color)
                    

            # Plot constraints
            rectangles = []
            for constraint in constraints:

                if constraint["variable"] == k_model:                

                    h = mpl.patches.Rectangle(
                        (constraint["timepoints"]["lb"], constraint["interval"]["lb"]),
                        constraint["timepoints"]["ub"] - constraint["timepoints"]["lb"],
                        constraint["interval"]["ub"] - constraint["interval"]["lb"]
                    )
                    rectangles.append(h)
            
            pc = mpl.collections.PatchCollection(
                rectangles, 
                facecolor = color, 
                alpha = 0.5,
                linewidth = 2,
                edgecolor = color
            )
            ax.add_collection(pc)

            __ = plt.setp(ax, ylabel = k_df)
        
        __ = plt.setp(fig.axes[-1], xlabel = "Timepoints")

        fig.savefig("./constraints.png", dpi = 150)

    return constraints

# %%
df = simulate_results
mapping_df_model = {"timepoint_unknown": "timepoints", "I_state": "I"}

constraints = generate_constraints_from_timeseries(
    model, 
    df, 
    mapping_df_model = mapping_df_model, 
    num_constraints = 2,
    plot = True
)

# %%
# Generate Funman request

r = generate_request(
    model,
    timepoints = list(range(0, 101, 10)),
    constraints = constraints,
    use_compartmental_constraints = True,
    normalization_constant = 1000,
    tolerance = 0.01, 
    parameters_of_interest = ["beta", "gamma"]
)

# %%
with open(os.path.join(PATH, "request.json"), "w") as fp:
    json.dump(r, fp, indent = 4)

# %%
