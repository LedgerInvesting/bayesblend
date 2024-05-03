from typing import Dict
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
from scipy.stats import gaussian_kde
import numpy as np

from .models import BayesBlendModel
from . import Draws

COLORS = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
]


def plot_blends(
    fits: Dict[str, BayesBlendModel], blends: Dict[str, Draws]
) -> go.Figure:
    """Plot the BayesBlend predictive distributions alongside
    the ELPD values for each candidate model and blend.

    This function takes a dictionary of fitted BayesBlendModel
    objects, which is used to extract the ELPD values of the
    candidate models, and a dictionary of Draws objects,
    which will most likely be the output of `BayesBlendModel.predict`.

    Args:
        fits: Dictionary of BayesBlendModel instances.
        blends: Dictionary of Draws instances.

    Returns:
        An instance of plotly.graph_objects.Figure,
        which can be used for future customization
        by the user.
    """
    R = len(blends)
    C = 2
    rc = [(r, c) for r in range(R) for c in range(C)]

    candidates = list(fits[next(iter(fits))].model_draws.keys())
    blenders = list(blends.keys())

    if not all(list(f.model_draws.keys()) == candidates for f in fits.values()):
        raise ValueError("All `fits` objects must contain the same models.")

    if any(f.lpd is None for fit in fits.values() for f in fit.model_draws.values()):
        raise ValueError("All `fits` must have valid log-likelihood values.")

    if any(blend.lpd is None for blend in blends.values()):
        raise ValueError("All `blends` must have valid log-likelihood values.")

    elpds = {
        k: (
            blend.lpd.sum(), # type: ignore
            elpd_se(blend.lpd),
        )
        for k, blend in blends.items()
    }
    elpds = elpds | {
        k: (
            f.lpd.sum(), # type: ignore
            elpd_se(f.lpd),
        )
        for k, f in fits[next(iter(fits))].model_draws.items()
    }

    COLOR_LOOKUP = {
        k: color
        for k, color in zip(
            blenders,
            COLORS,
        )
    }

    subplot_titles = [v for t in blenders for v in [f"{t} blend", f"{t} ELPD"]]

    fig = make_subplots(R, C, shared_xaxes="columns", subplot_titles=subplot_titles)
    fig.update_layout(template="none")

    for i, (r, c) in enumerate(rc):
        show_legend = False
        if c == 0:
            g, d = density(blends[blenders[r]].post_pred.flatten()) # type: ignore
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=g,
                    y=d,
                    marker_color=COLOR_LOOKUP[blenders[r]],
                    fill="tozeroy",
                    showlegend=show_legend,
                    name=blenders[r],
                ),
                row=r + 1,
                col=c + 1,
            )
            if r + 1 == R:
                fig.update_xaxes(title="y", row=r + 1, col=c + 1)
            fig.update_yaxes(title="Density", row=r + 1, col=c + 1)
        else:
            y_labels = [blenders[r], *candidates]
            elpd, se = zip(*[v for k, v in elpds.items() if k in y_labels])
            fig.add_trace(
                go.Scatter(
                    x=elpd,
                    y=y_labels,
                    mode="markers",
                    error_x={
                        "type": "data",
                        "array": se,
                        "width": 0,
                        "color": "gray",
                        "thickness": 0.75,
                    },
                    marker_color=[
                        COLOR_LOOKUP[blenders[r]],
                        *["gray"] * len(candidates),
                    ],
                    marker_symbol=["circle", *["circle-open"] * len(candidates)],
                    marker_size=12,
                    showlegend=show_legend,
                    name=blenders[r],
                ),
                row=r + 1,
                col=c + 1,
            )
            if r + 1 == R:
                fig.update_xaxes(title="ELPD", row=r + 1, col=c + 1)

    fig.update_xaxes(zeroline=False, col=2)
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(tickfont_size=15, col=1)
    fig.update_yaxes(tickfont_size=12, col=2)
    return fig


def elpd_se(lpd: np.ndarray | None) -> float:
    return lpd.std() * np.sqrt(len(lpd)) # type: ignore


def density(x: np.ndarray, steps: int = 500):
    limits = min(x), max(x)
    grid = np.linspace(*limits, steps)
    return grid, gaussian_kde(x)(grid)
