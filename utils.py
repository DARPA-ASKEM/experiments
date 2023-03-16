import plotly.graph_objs as go
import numpy as np


def get_best_period(frequency, power):
    best_frequency = frequency[np.argmax(power)]
    return 1.0 / best_frequency


def show_simple_scatter(x, y, title, xlabel, ylabel, dtick=None):
    scatter = go.Scatter(x=x, y=y)
    layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))
    fig = go.Figure(data=[scatter], layout=layout)

    if dtick not None:
        fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
            )
        )
    fig.show()


def check_for_periods(ls, min_period, max_period):
    max_frequency = 1.0 / min_period
    min_frequency = 1.0 / max_period
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency, maximum_frequency=max_frequency
        )
    best_period = get_best_period(frequency, power)
    prob = ls.false_alarm_probability(power.max())
    return frequency, power, best_period, prob, power.max()


def generate_intervals(start, end, interval_size):
    intervals = []
    for i in range(start, end, interval_size):
        intervals.append((i, i+interval_size))
    if intervals[-1][1] > end:
        intervals[-1] = (intervals[-1][0], end)
    return intervals
