import plotly.graph_objects as go

from bayesblend import plot_blends


def test_blended_score_plot(fit_models):
    model_names = (
        "pseudo_bma",
        "pseudo_bma+",
        "mle_stacking",
        "bayes_stacking",
        "hierarchical_bayes_stacking",
    )

    fit_dict = {k: f for k, f in zip(model_names, fit_models)}

    blend_dict = {k: f.predict() for k, f in zip(model_names, fit_models)}

    assert isinstance(plot_blends(fit_dict, blend_dict), go.Figure)
