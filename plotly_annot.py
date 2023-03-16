import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# generate example time series data
t = np.linspace(0, 10*np.pi, 1000)
y = np.sin(t)

# find peaks in the signal
peaks, _ = find_peaks(y)

# create a plotly figure
fig = go.Figure()

# add the time series data to the plot
fig.add_trace(go.Scatter(x=t, y=y, mode='lines'))

# add annotations marking the peaks
for peak in peaks:
    fig.add_annotation(x=t[peak], y=y[peak], text='Peak', showarrow=True,
                       arrowhead=1, ax=20, ay=-30)

# show the plot
fig.show()
