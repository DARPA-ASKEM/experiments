# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
# MITRE Starter Kit Epi Models
#
# 1. [J. Kantor](https://jckantor.github.io/CBE30338/03.09-COVID-19.html)

# %%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, Any
import json

# %%
# Kantor Models

def kantor_models(model_id: int = 1, params: Optional[list] = None, time: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:

    # params = [u, R0, t_incubation, t_infective, E_i, I_i, R_i]
    if params == None:
        params = [0.2, 2.4, 5.1, 3.3, 1.0 / 20000.0, 0.0, 0.0]

    u, R0, t_incubation, t_infective, E_i, I_i, R_i = params
    S_i = 1.0 - E_i - I_i - R_i
    alpha = 1.0 / t_incubation
    gamma = 1.0 / t_infective
    beta = R0 * gamma

    # Time
    if ~isinstance(time, np.ndarray):
        time = np.arange(0.0, 6.0 * 30.0, 0.5)

    # SIR ODE
    def SIR_ODE(x, t, u, alpha, beta, gamma):
        S, E, I, R = x
        dS = -(1 - u) * beta * S * I
        dE = 0.0
        dI = (1 - u) * beta * S * I - gamma * I
        dR =  gamma * I
        return [dS, dE, dI, dR]

    # SEIR ODE (Pan et al)
    def SEIR_ODE(x, t, u, alpha, beta, gamma):
        S, E, I, R = x
        dS = -(1 - u) * beta * S * I
        dE =  (1 - u) * beta * S * I - alpha * E
        dI = alpha * E - gamma * I
        dR =  gamma * I
        return [dS, dE, dI, dR]

    # High-Fidelity SEIR ODE (Boldog et al)
    def fSEIR_ODE(x, t, u, alpha, beta, gamma, mu):
        S, E1, E2, I1, I2, I3, R = x
        dS = -(1 - u) * beta * S * (I1 + I2 + I3)
        dE1 = - dS - 2 * alpha * E1
        dE2 = 2 * alpha * E1 - 2 * alpha * E2
        dI1 = 2 * alpha * E2 - 3 * gamma * I1 - mu * I1
        dI2 = 3 * gamma * I1 - 3 * gamma * I2 - mu * I2
        dI3 = 3 * gamma * I2 - 3 * gamma * I3 - mu * I3
        dR = 3 * gamma * I3
        return [dS, dE1, dE2, dI1, dI2, dI3, dR]

    # SIR
    if model_id == 0:

        u = 0.0
        alpha = None
        gamma = 1.0 / (t_incubation + t_infective)
        beta = R0 * gamma
        I_i = E_i
        E_i = None
        sol = sp.integrate.odeint(SIR_ODE, [S_i, E_i, I_i, R_i], time, args = (u, alpha, beta, gamma))
        S, E, I, R = sol.T
        
    # SEIR with intervention
    elif model_id == 1:
        sol = sp.integrate.odeint(SEIR_ODE, [S_i, E_i, I_i, R_i], time, args = (u, alpha, beta, gamma))
        S, E, I, R = sol.T

    # SEIR with intervention and fidelity
    elif model_id == 2:
        I_i = 0.0
        mu = 0.0
        sol = sp.integrate.odeint(fSEIR_ODE, [S_i, E_i, 0.0, 0.0, I_i, I_i, R_i], time, args = (u, alpha, beta, gamma, mu))
        S, E1, E2, I1, I2, I3, R = sol.T
        E = E1 + E2
        I = I1 + I2 + I3

    else:
        S = None
        E = None
        I = None
        R = None

    return {'params': params, 'time': time, 'S': S, 'E': E, 'I': I, 'R': R}

# %%
fig, axes = plt.subplots(1, 3, figsize = (10, 5))

for i, t in zip(range(3), ['Kantor.SIR', 'Kantor.SEIR.1', 'Kantor.SEIR.2']):

    sol = kantor_models(model_id = i)

    for k, v in sol.items():

        if k == 'time':
            time = v
        elif (k != 'params') & (~np.isnan(np.sum(v))):
            __ = axes[i].plot(time, v, label = k)
        else:
            pass
    
    __ = axes[i].set_title(t)
    __ = axes[i].legend()


fig.savefig('../figures/kantor_model_outputs.png', dpi = 150)

# %%
# Generate dataset of model outputs

# params = [u, R0, t_incubation, t_infective, E_i, I_i, R_i]
params = [0.2, 2.4, 5.1, 3.3, 1.0 / 20000.0, 0.0, 0.0]

# time: 6 months in 1-day steps
time = np.arange(0.0, 6.0 * 30.0, 1.0)

# u: [0.1, 0.3]
# R0: [2, 4]
# t_incubation: [2, 5]
# t_infective: [3, 6]

p = np.meshgrid(np.linspace(0.1, 0.3, 4), np.linspace(2, 4, 4), np.linspace(2, 5, 4), np.linspace(3, 6, 4))
for i, __ in enumerate(p):
    p[i] = p[i].reshape((-1, 1)).squeeze()

output = []
for i in range(len(p[0])):
    params = [p[0][i], p[1][i], p[2][i], p[3][i], 1.0 / 20000.0, 0.0, 0.0]
    sol = kantor_models(model_id = 1, time = time, params = params)
    output += [['kantor_SEIR_interv', '_'.join([f"{p:.1f}" for p in params]), k] + list(v) for k, v in sol.items() if k != 'params']

output = pd.DataFrame(output, columns = ['_model', '_scenario', '_varname'] + list(range(len(sol['time']))))

with open('../data/kantor_models.json', 'w') as f:
    f.write(output.to_json(orient = 'records'))

# %%




