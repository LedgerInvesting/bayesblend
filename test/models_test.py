import json
from functools import lru_cache
from typing import Dict

import numpy as np
import pytest
from cmdstanpy import CmdStanModel

from bayesblend import BayesStacking, HierarchicalBayesStacking, MleStacking, PseudoBma

STAN_FILE = "test/stan_files/bernoulli_ppc.stan"
DATA_FILE = "test/stan_data/bernoulli_data.json"

CFG: Dict = {"chains": 4, "parallel_chains": 4, "seed": 1234}


def compute_lpd(log_lik: np.ndarray) -> np.ndarray:
    max_ll = log_lik.max()
    return max_ll + np.log(np.mean(np.exp(log_lik - max_ll)))


with open(DATA_FILE, "r") as f:
    BERN_DATA = json.load(f)

MODEL = CmdStanModel(stan_file=STAN_FILE)
FIT1 = MODEL.sample(data=BERN_DATA, **CFG)
FIT2 = MODEL.sample(data=BERN_DATA | {"beta_alpha": 4, "beta_beta": 4}, **CFG)
FIT3 = MODEL.sample(data=BERN_DATA | {"beta_alpha": 10, "beta_beta": 2}, **CFG)

LPD = dict(
    fit1=np.apply_along_axis(compute_lpd, 0, FIT1.stan_variable("log_lik")),
    fit2=np.apply_along_axis(compute_lpd, 0, FIT2.stan_variable("log_lik")),
    fit3=np.apply_along_axis(compute_lpd, 0, FIT3.stan_variable("log_lik")),
)

# extreme LPD points to test that models can handle them
LPD_EXTREME = {
    fit: [-21000 if i == 0 else lpd for i, lpd in enumerate(lpd_points)]
    for fit, lpd_points in LPD.items()
}

DISCRETE_COVARIATES = {
    "dummy": ["group1"] * (BERN_DATA["N"] // 2) + ["group2"] * (BERN_DATA["N"] // 2)
}

CONTINUOUS_COVARIATES = {"metric": np.random.normal(size=BERN_DATA["N"])}

CONTINUOUS_COVARIATES_ZERO = {"metric_zero": np.zeros(BERN_DATA["N"])}


@lru_cache
def hierarchical_bayes_stacking():
    return HierarchicalBayesStacking(
        pointwise_diagnostics=LPD,
        discrete_covariates=DISCRETE_COVARIATES,
        seed=CFG["seed"],
    ).fit()


@lru_cache
def hierarchical_bayes_stacking_pooling():
    return HierarchicalBayesStacking(
        pointwise_diagnostics=LPD,
        discrete_covariates=DISCRETE_COVARIATES,
        partial_pooling=True,
        seed=CFG["seed"],
    ).fit()


def test_model_weights_valid():
    mle_stacking = MleStacking(pointwise_diagnostics=LPD).fit()
    bayes_stacking = BayesStacking(pointwise_diagnostics=LPD).fit()
    hier_bayes_stacking = hierarchical_bayes_stacking()
    pseudo_bma = PseudoBma(
        pointwise_diagnostics=LPD,
        bootstrap=False,
    ).fit()
    pseudo_bma_plus = PseudoBma(
        pointwise_diagnostics=LPD,
        n_boots=1000,
        seed=1234,
    ).fit()

    assert sum(mle_stacking.weights.values())
    assert sum(bayes_stacking.weights.values())
    assert all(sum(hier_bayes_stacking.weights.values())[0])
    assert sum(pseudo_bma.weights.values())
    assert sum(pseudo_bma_plus.weights.values())


def test_equal_diagnstics_equal_weights():
    lpd_dict = dict(model1=LPD["fit1"], model2=LPD["fit1"])
    stacking = MleStacking(pointwise_diagnostics=lpd_dict).fit()
    assert all([w == 0.5 for w in stacking.weights.values()])


def test_bayes_stacking_weight_extreme_elpd():
    mle_stacking = MleStacking(pointwise_diagnostics=LPD_EXTREME).fit()
    bayes_stacking = BayesStacking(pointwise_diagnostics=LPD_EXTREME).fit()
    # MLE optimization will fail to converge, Bayes will succeed
    assert not mle_stacking.model_info.success
    assert bayes_stacking.model_info.summary()["R_hat"].max() < 1.01


def test_bayes_stacking_weight_arrays():
    bayes_stacking = BayesStacking(pointwise_diagnostics=LPD).fit()
    hier_bayes_stacking = hierarchical_bayes_stacking()
    # internal weights are full posteriors, and should always have shape[0]>1
    # hierarchical stacking should also have weights for each datapoint, or shape[1]==N
    assert all(v.shape[0] > 1 for v in bayes_stacking._weights.values())
    assert all(v.shape[1] == 1 for v in bayes_stacking._weights.values())
    assert all(v.shape[0] > 1 for v in hier_bayes_stacking._weights.values())
    assert all(
        v.shape[1] == BERN_DATA["N"] for v in hier_bayes_stacking._weights.values()
    )

    # external/user-readable weights should be posterior means, reducing to shape[0]==1
    # Otherwise, dimenions should be same as above
    assert all(v.shape[0] == 1 for v in bayes_stacking.weights.values())
    assert all(v.shape[1] == 1 for v in bayes_stacking.weights.values())
    assert all(v.shape[0] == 1 for v in hier_bayes_stacking.weights.values())
    assert all(
        v.shape[1] == BERN_DATA["N"] for v in hier_bayes_stacking.weights.values()
    )


def test_hier_bayes_stacking_no_covariates_fails():
    with pytest.raises(ValueError):
        HierarchicalBayesStacking(
            pointwise_diagnostics=LPD,
        )


def test_hier_bayes_stacking_only_continuous_covariates():
    hier_bayes_stacking = HierarchicalBayesStacking(
        pointwise_diagnostics=LPD,
        continuous_covariates=CONTINUOUS_COVARIATES,
        cmdstan_control=CFG,
    ).fit()

    assert hier_bayes_stacking.predict(continuous_covariates=CONTINUOUS_COVARIATES)


def test_hier_bayes_stacking_predict_different_covariates_fails():
    hier_bayes_stacking = hierarchical_bayes_stacking()
    different_covariates = {"fail": ["group1"] * 2 + ["group2"] * 2}
    with pytest.raises(ValueError):
        hier_bayes_stacking.predict(discrete_covariates=different_covariates)


def test_hier_bayes_stacking_predict_different_covariate_levels():
    hier_bayes_stacking = hierarchical_bayes_stacking()
    # covariate name is the same, but there is a new level
    new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    with pytest.raises(ValueError):
        hier_bayes_stacking.predict(discrete_covariates=new_level)

    # covariate name is the same, but there is a missing level
    # should generate predictions as normal without failing
    missing_level = {"dummy": ["group1"] * 4}
    hier_bayes_stacking.predict(discrete_covariates=missing_level)


def test_hier_bayes_stacking_pooling():
    hier_bayes_stacking_pooled = hierarchical_bayes_stacking_pooling()

    hyperprior_pars = ["mu_global"]
    prior_pars = ["mu_disc", "sigma_disc"]

    n_models = len(LPD)
    model_coefs = hier_bayes_stacking_pooled.coefficients

    assert all(
        par in hier_bayes_stacking_pooled.coefficients
        for par in hyperprior_pars + prior_pars
    )
    # check shape of model hyperprior parameters
    assert all(model_coefs[par].shape == (1, 1) for par in hyperprior_pars)
    # check shape of model prior parameters
    assert all(model_coefs[par].shape == (1, n_models - 1) for par in prior_pars)


def test_hier_bayes_stacking_predict_different_covariate_levels_pooling():
    hier_bayes_stacking = hierarchical_bayes_stacking_pooling()
    # covariate name is the same, but there is a new level
    one_new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    with pytest.warns(UserWarning):
        hier_bayes_stacking.predict(discrete_covariates=one_new_level)

    two_new_levels = {"dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2}
    with pytest.warns(UserWarning):
        hier_bayes_stacking.predict(discrete_covariates=two_new_levels)

    three_new_levels = {
        "dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2 + ["group97"] * 2
    }
    with pytest.warns(UserWarning):
        hier_bayes_stacking.predict(discrete_covariates=three_new_levels)


def test_hier_bayes_stacking_continuous_covariates_transform():
    transforms = ["identity", "standardize", "relu"]
    stacks = {}
    for transform in transforms:
        stacks[transform] = HierarchicalBayesStacking(
            pointwise_diagnostics=LPD,
            continuous_covariates=CONTINUOUS_COVARIATES,
            continuous_covariates_transform=transform,
            cmdstan_control=CFG,
        ).fit()

    assert all(k == v.continuous_covariates_transform for k, v in stacks.items())
    assert len(stacks["relu"].coefficients["beta_cont"][0]) == 2

    with pytest.raises(ValueError, match="logit not found"):
        HierarchicalBayesStacking(
            pointwise_diagnostics=LPD,
            continuous_covariates=CONTINUOUS_COVARIATES,
            continuous_covariates_transform="logit",
            cmdstan_control=CFG,
        ).fit()

    # error if trying to standardize variable with 0 SD/variance
    with pytest.raises(ValueError, match="cannot be standardized"):
        HierarchicalBayesStacking(
            pointwise_diagnostics=LPD,
            continuous_covariates=CONTINUOUS_COVARIATES_ZERO,
            continuous_covariates_transform="standardize",
            cmdstan_control=CFG,
        ).fit()


def test_hier_bayes_stacking_weight_predictions():
    hier_bayes_stacking = hierarchical_bayes_stacking()
    weight_predictions_old_data = hier_bayes_stacking.predict(
        discrete_covariates=DISCRETE_COVARIATES
    )
    new_covariates = {"dummy": ["group1"] * 2 + ["group2"] * 2}
    weight_predictions_new_data = hier_bayes_stacking.predict(
        discrete_covariates=new_covariates
    )

    # predictions from passing in same data should be approx equal to estimated weights
    assert all(
        np.allclose(true, pred, atol=0.001)
        for true, pred in zip(
            hier_bayes_stacking.weights.values(), weight_predictions_old_data.values()
        )
    )
    # prediction first dimension should be 1
    assert all([v.shape[0] == 1 for v in weight_predictions_new_data.values()])
    # prediction second dimension should match length of covariate vector entered
    assert all(
        [
            v.shape[1] == len(new_covariates["dummy"])
            for v in weight_predictions_new_data.values()
        ]
    )


def test_hier_bayes_stacking_predictions_new_level_dummy_codes():
    hier_bayes_stacking = hierarchical_bayes_stacking_pooling()
    # covariate name is the same, but there are new levels
    new_levels = {"dummy": ["group2"] * 5 + ["group10"] * 5}

    # set the group-level covariate coefficent to all 0s to force
    # new level coefficent to be exactly 0. The estimated weights
    # will then be the same as the predicted weights. This ensures
    # that the dummy coding for new levels is working as intended
    hier_bayes_stacking._coefficients["mu_disc"] = np.zeros_like(
        hier_bayes_stacking._coefficients["mu_disc"]
    )
    hier_bayes_stacking._coefficients["sigma_disc"] = np.zeros_like(
        hier_bayes_stacking._coefficients["sigma_disc"]
    )
    predicted_weights = hier_bayes_stacking.predict(discrete_covariates=new_levels)

    # predicitons where dummy=group10 should be the same as when dummy=group1,
    # and predictions for dummy=group2 should remain the same as before. Because
    # original weights are ordered per DISCRETE_COVARIATES, predictions should
    # match reversed original weights
    assert all(
        np.allclose(true[:, ::-1], pred, atol=0.001)
        for true, pred in zip(
            hier_bayes_stacking.weights.values(), predicted_weights.values()
        )
    )


def test_bayes_stacking_generation_with_priors():
    hier_priors = {
        "alpha_loc": 1,
        "alpha_scale": 2,
        "beta_disc_loc": 1,
        "beta_disc_scale": 2,
        "beta_cont_loc": 1,
        "beta_cont_scale": 2,
        "tau_mu_global": 2,
        "tau_mu_disc": 2,
        "tau_mu_cont": 2,
        "tau_sigma_disc": 2,
        "tau_sigma_cont": 2,
        "lambda_loc": 0.5,
    }

    bayes_stacking = BayesStacking(LPD, cmdstan_control=CFG).fit()
    bayes_stacking_priors = BayesStacking(
        LPD, cmdstan_control=CFG, priors={"w_prior": [2, 2, 2]}
    ).fit()
    hier_bayes_stacking = HierarchicalBayesStacking(
        LPD, cmdstan_control=CFG, discrete_covariates=DISCRETE_COVARIATES
    ).fit()
    hier_bayes_stacking_priors = HierarchicalBayesStacking(
        LPD,
        cmdstan_control=CFG,
        discrete_covariates=DISCRETE_COVARIATES,
        priors=hier_priors,
    ).fit()

    assert bayes_stacking.priors == {"w_prior": [1, 1, 1]}
    assert bayes_stacking_priors.priors == {"w_prior": [2, 2, 2]}
    assert hier_bayes_stacking.priors == HierarchicalBayesStacking.DEFAULT_PRIORS
    assert hier_bayes_stacking_priors.priors == hier_priors

    with pytest.raises(ValueError):
        BayesStacking(LPD, priors={"w_prior": [1]})

    with pytest.raises(ValueError):
        HierarchicalBayesStacking(
            LPD,
            discrete_covariates=DISCRETE_COVARIATES,
            priors=hier_priors | {"x": 1},
        )


def test_bayes_stacking_adaptive_prior_structure():
    base = hierarchical_bayes_stacking()
    adaptive = HierarchicalBayesStacking(
        pointwise_diagnostics=LPD,
        discrete_covariates=DISCRETE_COVARIATES,
        adaptive=True,
        cmdstan_control=CFG,
    ).fit()

    _lambda = adaptive.model_info.stan_variable("lambda")
    assert adaptive.adaptive
    assert not base.adaptive
    assert "lambda" in adaptive.model_info.stan_variables().keys()
    assert "lambda" not in base.model_info.stan_variables().keys()
    assert np.isclose(_lambda.mean(), 0.25, rtol=1e-1)
