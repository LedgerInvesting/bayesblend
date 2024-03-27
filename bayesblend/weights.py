from typing import Dict, Literal, Optional

import numpy as np
from cmdstanpy import CmdStanMCMC

from .models import (
    BayesBlendModel,
    BayesStacking,
    HierarchicalBayesStacking,
    MleStacking,
    PseudoBma,
)

WEIGHT_METHODS_TYPE = Literal[
    "mle_stacking", "bayes_stacking", "hier_bayes_stacking", "pseudo_bma"
]

WEIGHT_METHODS = {
    "mle_stacking": MleStacking,
    "bayes_stacking": BayesStacking,
    "hier_bayes_stacking": HierarchicalBayesStacking,
    "pseudo_bma": PseudoBma,
}

INFORMATION_CRITERIA = ["elpd_lfo", "elpd_loo", "elpd_waic"]


def model_weights(
    fits: Optional[Dict[str, CmdStanMCMC]] = None,
    log_densities: Optional[Dict[str, np.ndarray]] = None,
    method: WEIGHT_METHODS_TYPE = "mle_stacking",
    information_criteria: Optional[str] = None,
    lfo_dict: Optional[Dict[str, Dict]] = None,
    **kwargs,
) -> BayesBlendModel:
    """Compute a set of weights across candidate models using a variety of methods.

    Stacking, the default option, uses Bayesian stacking by optimizing the set of
    weights across candidate models that maximizes the log score of the data given the
    model predictions.

    Information criteria weights are derived by a simple rescaling procedure
    (computing the differences between each IC and the maximum IC) and running the
    rescaled values through a softmax function. This procedure is referred to as
    pseudo Bayesian model averaging (BMA), whereas traditional BMA weights models
    by their marginal likelihoods (the denominator in Bayes' rule). However, the
    marginal likelihood is non-trivial to calculate from most models.

    The `bootstrap` option allows computing pseudo-BMA+ weights, which account for the
    uncertainty in information criteria by using a Bayesian bootstrap procedure to
    compute the distribution of log scores from the approximate leave-one-out predictive
    densities. The average weight across bootstrap replicates is used for each model.
    This is the default procedure because it performs better in so-called M-complete and M-open
    contexts.

    For further information, see Yao et al. (2018):
    http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf

    Args:
        fits: a dictionary of fit names and AutofitModel instances. Can be None if
            `information_criteria` = `elpd_lfo`.
        log_densities: a dictionary of fit names and LPD points. Can be None if
            using AutofitModel instances or LFO fits.
        method: the method to calculate the weights. Defaults to `stacking`.
        information_criteria: which information criteria to use to calculate the
            weights. Defaults to None.
        lfo_dict: a dictionary of names and leave future out cross-validation estimates derived from Dunsink's lfo routine.
        kwargs: keyword arguments to the weight method. See the weight method functions below.

    """

    if method not in WEIGHT_METHODS:
        raise ValueError(f"{method} is an unknown weighting method.")

    # compute the pointwise diagnostics that weight methods depend on
    if log_densities is not None:
        n_data_points = [len(v) for v in log_densities.values()]
        if any([n != n_data_points[0] for n in n_data_points]):
            raise ValueError(
                "Cannot compare models with different number of data points."
            )
        pointwise_diagnostics = {k: {"lpd_points": v} for k, v in log_densities.items()}

    else:
        if (
            information_criteria is not None
            and information_criteria.lower() not in INFORMATION_CRITERIA
        ):
            raise ValueError(
                f"`information_criteria` not recognized; must be one of {INFORMATION_CRITERIA}"
            )
        if (
            information_criteria is not None
            and information_criteria.lower() != "elpd_lfo"
        ):
            if fits is None:
                raise ValueError(
                    "Must supply fitted model objects when using elpd_loo or elpd_waic"
                )
            diagnostics = {k: v.diagnostics() for k, v in fits.items()}
            if len(diagnostics[next(iter(diagnostics))].log_likelihood) > 1:
                raise ValueError(
                    "Cannot currently pass AutofitModel objects with multiple likelihoods "
                    "to dunsink.weights."
                )
            ll_name = list(diagnostics[next(iter(diagnostics))].log_likelihood)[0]
            pointwise_diagnostics = {
                k: v.log_likelihood[ll_name][information_criteria.lower()]
                for k, v in diagnostics.items()
            }
        else:
            if lfo_dict is None:
                raise ValueError(
                    "One of `log_densities`, `information_criteria`, or `lfo_dict` must be "
                    "defined to estimate weights."
                )
            pointwise_diagnostics = {}
            for k, v in lfo_dict.items():
                pointwise_diagnostics[k]["estimate"] = lfo_dict[k]["elpd"]
                pointwise_diagnostics[k]["lpd_points"] = lfo_dict[k]["elpd_i"]

    fit = WEIGHT_METHODS[method](pointwise_diagnostics, **kwargs).fit()
    return fit
