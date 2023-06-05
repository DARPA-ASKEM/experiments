# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Thin-Thread TA3 Integration Work for July 2023 Hackathon
#
# 1. Extract subset of January 2023 Hackathon evaluation dataset that can be used for model calibration

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# 1. Subset dataset for model calibration

df = pd.read_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-IRDVHN_age.csv', index_col = 0)
df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')

N = df[[c for c in df if c[0] == 'N']].sum(axis = 1)
not_S = df[[c for c in df if c in ('I', 'R', 'H')]].sum(axis = 1)
df['S'] = N - not_S

# %%

m = {
    'S': 'Susceptible',
    'I': 'Infected',
    'R': 'Recovered',
    'D': 'Dead',
    'V': 'Vaccinated',
    'H': 'Hospitalized'
}

t = range(380, 420)
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
for c in df:
    if len(c) < 2:
        # __ = ax.plot(df['date'], df[c], label = m[c])
        # __ = ax.plot(range(df.shape[0]), df[c], label = m[c])
        __ = ax.plot(t, df[c].iloc[t], label = m[c])

__ = ax.legend()
__ = plt.setp(ax, yscale = 'log')

df.iloc[t, :].to_csv('../../thin-thread-examples/milestone_6month/evaluation/ta1/usa-IRDVHN_age_subset.csv', index = False)

# %%