import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from uxils.plot.plotly_ext import show_figure, update_fig

pallete = px.colors.qualitative.Pastel

with open("measurements_svd_(10000, 4000)_32.pkl", "rb") as in_file:
    g_res = pickle.load(in_file)

fig = go.Figure()

# color = pallete[idx]
x = np.array(list(g_res.keys()))
y = np.array(list(g_res.values()))
print(x, y)

fig.add_trace(
    go.Bar(
        x=x,
        y=y,
        marker={'color': pallete}
    )
)

title = "Truncated SVD of dense 10000 x 4000 matrix; 32 largest singular values"
update_fig(
    fig,
    xaxis_title="Method",
    yaxis_title="time, seconds",
    legend_title="Used method:",
    title=title,
    font_size=30,
)
show_figure(fig)
