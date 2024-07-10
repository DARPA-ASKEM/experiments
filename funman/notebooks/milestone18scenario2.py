# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # 18-Month Milestone Epi Evaluation Scenario 2
#
# 1. Start with a base SIRHRD model
# 2. Update it with a one-dose vaccination process (to support interventions)
# 3. Model checks, (1) conservation of total pop, (2) total unvaccinated pop is non-increasing, (3) total vaccinated pop is non-decreasing, (4) ???
# 4. Update it again with time-varying testing
# 5. Stratify with 4 different strata (1/2-dose vaccination, age, sex, ethnicity)
# 6. Model checks again, same checks

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

import sympy
from mira.metamodel import *
from mira.modeling.viz import GraphicalModel
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%[markdown]
# ## Load models

# %%
MODEL_PATH = "./data/milestone18_evaluation/scenario2"
model_names = {
    "base": "SIRHD_base.json", 
    "vax1": "scenario2_q4_petrinet.json", 
    "base_testing": "SIRHD_base_testing.json", 
    "base_testing_vax2": "SIRHD_base_testing_multivax.json"
}

models = {}
for name, filename in model_names.items():
    with open(os.path.join(MODEL_PATH, filename), "r") as f:
        models[name] = json.load(f)

# %%
# Visualize
tm = template_model_from_amr_json(models["base"])
GraphicalModel.for_jupyter(tm)

# %%
def generate_parameter_table(model: dict) -> pd.DataFrame:

    data = {k: [] for k in ("id", "value", "dist", "lb", "ub")}
    for p in model["semantics"]["ode"]["parameters"]:

        data["id"].append(p["id"])
        data["value"].append(p["value"])
        if "distribution" in p.keys():
            data["dist"].append(p["distribution"]["type"])
            data["dist"].append(p["distribution"]["parameters"]["minimum"])
            data["dist"].append(p["distribution"]["parameters"]["maximum"])
        else:
            data["dist"].append(None)
            data["lb"].append(None)
            data["ub"].append(None)

    return pd.DataFrame(data)

# %%
# Construct Funman request
def generate_request(model: dict, timepoints: list, constraints: list[dict] = [], use_compartmental_constraints: bool = True, tolerance: float = 0.1, parameters_of_interest: list[str] = []) -> dict:

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
                "normalization_constant": 1,
                "tolerance": tolerance
            }
        }
    }

    return request

# %%
# r = generate_request(
#     models["base"],
#     timepoints = list(range(0, 101, 10)),
#     checks = {"I": []}
# )

# %%

