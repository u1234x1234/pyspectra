import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from uxils.plot.plotly_ext import show_figure, update_fig

pallete = px.colors.qualitative.G10

with open("measurements2.pkl", "rb") as in_file:
    g_res = pickle.load(in_file)

fig = go.Figure()

for idx, (name, values) in enumerate(g_res.items()):
    x = np.array(list(values.keys()))
    y = np.array(list(values.values())).mean(axis=1)

    color = pallete[idx]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            hoveron="points",
            name=name,
            line=dict(color=color),
        )
    )

title = f"Finding {5} largest eigen[values]vectors of symmetric dense matrix"
update_fig(
    fig,
    xaxis_title="Size of matrix",
    yaxis_title="time, seconds",
    legend_title="used method",
    title=title,
    x_log=False,
)
show_figure(fig)
