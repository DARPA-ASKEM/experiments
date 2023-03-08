import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft
from statsmodels.tsa.seasonal import STL
import os
import requests
import urllib.request
from matplotlib import pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go

from astropy.timeseries import LombScargle
from utils import get_best_period, show_simple_scatter, check_for_periods, generate_intervals


if not os.path.exists("owid-covid-data.csv"):
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    r = requests.get(url, allow_redirects=True)
    with open("owid-covid-data.csv", "wb") as f:
        f.write(r.content)


df = pd.read_csv("owid-covid-data.csv")

# create new dataframe with just columns: "date", "new_cases", "new_deaths", where "location" is "United States"
us_data = df.loc[df["location"] == "United States", ["date", "new_cases", "new_deaths"]]

# replace any nans with 0
us_data = us_data.fillna(0)

us_data["date"]


# This is our basic data we want to understand
y = us_data["new_cases"].to_numpy()
dates = pd.to_datetime(us_data["date"]).to_numpy()



# def get_best_period(frequency, power):
#     best_frequency = frequency[np.argmax(power)]
#     return 1.0 / best_frequency


# def show_simple_scatter(x, y, title, xlabel, ylabel):
#     scatter = go.Scatter(x=x, y=y)
#     layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))
#     fig = go.Figure(data=[scatter], layout=layout)
#     fig.show()


# Here we create an interactive scatter plot

# new_cases_scatter = go.Scatter(x=dates, y=y)
# layout = go.Layout(title='New Covid Cases', xaxis=dict(title='Date'),
#                    yaxis=dict(title='(New Cases)'))
# fig = go.Figure(data=[new_cases_scatter], layout=layout)
# fig.show()
show_simple_scatter(dates, y, "New Covid Cases", "Date", "New Cases")

# To use LombScargle y needs to be in units of seconds,
# minutes, hours, days, years.  We are looking to get frequency
# in cycles per unit time.  Not angular frequencies.


# The following ndarrays have float64 type
times = (dates - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
hours = times / 3600
days = hours / 24

first_day = days[0]

days = days - first_day

# norm_y = (y - np.mean(y)) / (np.max(y) - np.min(y)) * 2  #TODO should I use this one for LS as well?
# z = np.abs(fft(norm_y))
# peaks, _ = find_peaks(z, prominence=10) # TODO how do we pick prominence??

# Now we need need to generate intervals around these
# peaks = [   1,    3,    6,    9,  162,  324,  486,  647,  809,  971, 1124, 1127, 1130]  These last two points can be thrown out because they are at the end of the range
# period_ranges = [(0, 2), (2, 5), (4, 8), (5, 20), (100, 200), (200, 300)]
period_ranges = [(1, 2), (2, 5), (4, 8), (7, 100), (100, 200), (200, 400), (400, 600), (700, 1000), (1000, 1125)]

show_simple_scatter(days, y, "New Covid Cases", "day", "New Cases")
show_simple_scatter(days, z, "FFT", "day", "frequency")
ls = LombScargle(days, y)

# frequency, power, best_period, prob = check_for_periods(ls, 1, 1132)
# sorted_indices = np.argsort(power)[::-1]
# sorted_freqs = frequency[sorted_indices]
# sorted_power = power[sorted_indices]
# sorted_probs = [ls.false_alarm_probability(p) for p in sorted_power]

# temp = [(i,j,k) for i, j, k in zip(1.0 / sorted_freqs, sorted_power, sorted_probs) if j > 0.1]

# for i,j,k in zip(1.0 / sorted_freqs, sorted_power, sorted_probs):
#     print(i, j, k)

# There are only 1132 days.  Do we throw out the very end of the FT spectrum?
# peaks = [   1,    3,    6,    9,  162,  324,  486,  647,  809,  971, 1124, 1127, 1130]
# Below approach of picking edges of 
# period_ranges = [(1, 3), (1, 6), (3, 9), (6, 162), (162, 486), (324, 647), (647, 971), (809, 1124), (1124, 1130), (1127, 1400)]
period_ranges = [(1, 2), (2, 5), (4, 8), (7, 100), (100, 200), (200, 400), (400, 600), (700, 1000), (1000, 1125)]

for i, j in period_ranges:
    frequency, power, best_period, prob, max_power = check_for_periods(ls, i, j)
    print("Period of likely cyclicality ", best_period ) # Convert to days
    print("probability of a false alarm ", prob)
    print("Significance of signal ", max_power)
    print()


frequency, power, best_period, prob, max_power = check_for_periods(ls, 1, 350)
# period_ranges = [(1, 5),(5,10), (10, 20), (20, 30), (30, 40), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]

# period_ranges = generate_intervals(1, int(days[-1]), 3)

# for i, j in period_ranges:
#     frequency, power, best_period, prob = check_for_periods(ls, i, j)
#     print("Period of likely cyclicality ", best_period) # Convert to days
#     print("probability of a false alarm ", prob)
#     print()

# frequency, power, best_period, prob = check_for_periods(ls, 0.1, 10)
# print("Period of likely cyclicality ", best_period)
# print("probability of a false alarm ", prob)


# frequency, power = ls.autopower(minimum_frequency=0.02, maximum_frequency=1.0)

# print("Best period: ", get_best_period(frequency, power))

# # Create a new scatter plot for this
# show_simple_scatter(1.0 / frequency, power, "Lomb Scargle", "period (days)", "Power")
