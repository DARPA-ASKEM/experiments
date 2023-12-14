# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# SIDARTHE Configuration Examples with Piecewise Parameters
# 
# The SIDARTHE paper describes nine scenarios where the model parameters
# change over time. Example AMR JSONs have been manually created to 
# capture this dynamics using placeholder list of timepoint-value pairs. 
# To comform to the PetriNet schema, they need to be converted to 
# piecewise-defined rate laws.
#
# semantics.ode.parameters[].value = [[timepoint, value], ...]
# ==>
# new parameter for each value (p -> p0, p1, ...)
# p -> p(t) = p0 * Heaviside(t - t0, 1) + (p1 - p0) * Heaviside(t - t1, 1) + ...

# %%
import json
import copy
import os
from tqdm import tqdm 

# %%
PATH = "../data/configure_model_from_docs"

# %%
for f in tqdm(os.listdir(PATH + "/example_2")):

    n, ext = f.split(".")

    if (n != "input_model") & (ext == "json"):
        
        with open(os.path.join(PATH, "example_2", f), "r") as h:
            model = json.load(h)

        param_dict = {p["id"]: p["value"] for p in model["semantics"]["ode"]["parameters"]}

        # Define new parameters from each timepoint-value pair
        params_new = [p for p in model["semantics"]["ode"]["parameters"] if isinstance(p["value"], list) == False]
        for p, vals in param_dict.items():
            if isinstance(vals, list):
                for i, (t, v) in enumerate(vals):
                    params_new.append({"id": f"{p}_{i}", "value": v})
                    params_new.append({"id": f"t_{p}_{i}", "value": t})

        # Generate sympy expression of p(t)
        # p(t) = p_0 * H(t - t_p_0) + (p_1 * p_0) * H(t - t_p_1) + ...
        param_expr = {p: "" for p, vals in param_dict.items() if isinstance(vals, list)}
        for p, vals in param_dict.items():
            if isinstance(vals, list):
                for i, (t, v) in enumerate(vals):
                    if i == 0:
                        param_expr[p] += f"{p}_{i} * Heaviside(t - t_{p}_{i}, 1)"
                    else:
                        param_expr[p] += f" + ({p}_{i} - {p}_{i - 1}) * Heaviside(t - t_{p}_{i}, 1)"

        # Map all p in rates to p(t)
        rates_new = []
        for r in model["semantics"]["ode"]["rates"]:
            r_new = r["expression"]
            for p, expr in param_expr.items():
                if r_new.find(f"*{p}") != -1:
                    r_new = r_new.replace(p, f"({expr})")
            rates_new.append({"target": r["target"], "expression": r_new})

        # Replacement model
        model_p = copy.deepcopy(model)
        model_p["semantics"]["ode"]["parameters"] = params_new
        model_p["semantics"]["ode"]["rates"] = rates_new

    elif f == "input_model.json":
        with open(os.path.join(PATH, "example_2", f), "r") as h:
            model_p = json.load(h)
    else:
        pass

    # Save
    with open(os.path.join(PATH, "example_3", f), "w") as f:
        json.dump(model_p, f, indent = 2)

# %%
