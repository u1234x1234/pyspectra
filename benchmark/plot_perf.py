import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from uxils.plot.plotly_ext import show_figure, update_fig

pallete = px.colors.qualitative.G10

with open("measurements_dense_16.pkl", "rb") as in_file:
    g_res = pickle.load(in_file)

fig = go.Figure()

for idx, (name, values) in enumerate(g_res.items()):
    if "numpy" in name:
        continue

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

title = "Top-16 eigenvalues of a symmetric dense matrix"
update_fig(
    fig,
    xaxis_title="Matrix size",
    yaxis_title="time, seconds",
    legend_title="Used method:",
    title=title,
    x_log=False,
    font_size=30,
)
show_figure(fig)
