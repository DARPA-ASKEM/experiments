import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

def get_best_period(frequency, power):
    best_frequency = frequency[np.argmax(power)]
    return 1.0 / best_frequency


def show_simple_scatter(x, y, title, xlabel, ylabel):
    scatter = go.Scatter(x=x, y=y)
    layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()