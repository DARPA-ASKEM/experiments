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

from utils import get_best_period, show_simple_scatter


tides_data = pd.read_csv("Tide Prediction.csv")

heights_df = tides_data.loc[
    tides_data["stationID"] == "Aranmore", ["time", "Water_Level"]
]

times = (heights_df['time'] - pd.Timestamp("1970-01-01T00:00:00Z"))
seconds = np.asarray([t.total_seconds() for t in times])
minutes = seconds / 60.0

first_minute = minutes[0]

# minutes and y are now np.float64
minutes = minutes - first_minute
y = heights_df["Water_Level"].to_numpy().astype(float)

show_simple_scatter(heights_df["time"], y, "Tides vs data time", "datetime", "height")

show_simple_scatter(minutes, y, "Tides", "minutes", "height")

ls = LombScargle(minutes, y)

# Below gives period of 0.52 days
frequency, power = ls.autopower(minimum_frequency=0.0001, maximum_frequency=0.005)
print("Best period: ", get_best_period(frequency, power) / 60.0 / 24.0)
print(ls.false_alarm_probability(power.max()))
show_simple_scatter(frequency, power, "Lomb Scargle for Tides Data 0.", "frequency", "Power")
show_simple_scatter(1.0 / frequency / 60.0 / 24.0, power, "Lomb Scargle for Tides Data", "Period (days)", "Power")

# Below gives a period of 31.9 days
frequency, power = ls.autopower(minimum_frequency=0.00001, maximum_frequency=0.0005)
print("Best period: ", get_best_period(frequency, power) / 60.0 / 24.0)
print(ls.false_alarm_probability(power.max()))
show_simple_scatter(frequency, power, "Lomb Scargle for Tides Data 0.", "frequency", "Power")
show_simple_scatter(1.0 / frequency / 60.0 / 24.0, power, "Lomb Scargle for Tides Data", "Period (days)", "Power")

# Below gives a period of 368 days
frequency, power = ls.autopower(minimum_frequency=0.000001, maximum_frequency=0.00001)
print("Best period: ", get_best_period(frequency, power) / 60.0 / 24.0)
print(ls.false_alarm_probability(power.max()))
show_simple_scatter(frequency, power, "Lomb Scargle for Tides Data 0.", "frequency", "Power")
show_simple_scatter(1.0 / frequency / 60.0 / 24.0, power, "Lomb Scargle for Tides Data", "Period (days)", "Power")
