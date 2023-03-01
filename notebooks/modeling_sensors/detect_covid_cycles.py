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


def get_best_period(frequency, power):
    best_frequency = frequency[np.argmax(power)]
    return 1.0 / best_frequency


def show_simple_scatter(x, y, title, xlabel, ylabel):
    scatter = go.Scatter(x=x, y=y)
    layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


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
show_simple_scatter(days, y, "New Covid Cases", "day", "New Cases")

ls = LombScargle(days, y)

# The minimum period should be 0 cases per day
# the maximum period
frequency, power = ls.autopower()

print("Best period: ", get_best_period(frequency, power))

# Create a new scatter plot for this
show_simple_scatter(frequency, power, "Lomb Scargle", "cycles per day", "Power")
