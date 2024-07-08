# Maybe this link https://dash.plotly.com/basic-callbacks can help organize sharable hover.
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.light_gcot import LightGCOT


def plot_A_parameters(model: LightGCOT, log: bool = False) -> dict[str, go.Figure] | None:
    # Initialize figure with subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=(r"$\log{w_n}$", r"$a_n$", r"$A_n$"))

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=np.arange(model.n_potentials),
            y=model.log_w.cpu().detach().numpy(),
            marker=dict(color="crimson"),
            mode="markers",
            customdata=np.arange(model.n_potentials),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=model.a[:, 0].cpu().detach().numpy(),
            y=model.a[:, 1].cpu().detach().numpy(),
            marker=dict(color="crimson"),
            mode="markers",
            customdata=np.arange(model.n_potentials),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=model.A_diagonal_matrix[:, 0].cpu().detach().numpy(),
            y=model.A_diagonal_matrix[:, 1].cpu().detach().numpy(),
            marker=dict(color="crimson"),
            mode="markers",
            customdata=np.arange(model.n_potentials),
        ),
        row=1,
        col=3,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=3)

    # Update title and height
    fig.update_layout(
        title_text="A parameters",
        title_x=0.5,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey")
    fig.update_yaxes(mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey")

    # Update trace
    fig.update_traces(hovertemplate="%{customdata}: (%{x:,.4f}, %{y:,.4f})<extra></extra>")

    if log:
        A_dict = {"A parameters": fig}
        return A_dict
    else:
        fig.show()
