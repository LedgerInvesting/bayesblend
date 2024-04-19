import copy
import json
from functools import lru_cache

import arviz as az
import numpy as np
import pytest
from cmdstanpy import CmdStanModel

from bayesblend import (
    BayesStacking,
    HierarchicalBayesStacking,
    MleStacking,
    PseudoBma,
    SimpleBlend,
)
from bayesblend.io import Draws

STAN_FILE = "test/stan_files/bernoulli_ppc.stan"
DATA_FILE = "test/stan_data/bernoulli_data.json"

MODEL = CmdStanModel(stan_file=STAN_FILE)

CFG = {"chains": 4, "parallel_chains": 4}

SEED = 12344

with open(DATA_FILE, "r") as f:
    BERN_DATA = json.load(f)

FIT = MODEL.sample(data=BERN_DATA, **CFG)

RNG = np.random.RandomState(SEED)


def make_draws(mu, p, n_samples=1000, n_datapoints=10, shape=None):
    shape = (n_samples, n_datapoints) if shape is None else shape
    return Draws(
        log_lik=np.array(
            [RNG.normal(mu, 0.1, n_samples) for _ in range(n_datapoints)]
        ).T.reshape(shape),
        post_pred=np.array(
            [RNG.choice([0, 1], size=n_samples, p=p) for _ in range(n_datapoints)]
        ).T.reshape(shape),
    )


MODEL_DRAWS = {
    "fit1": make_draws(-1, [0.9, 0.1]),
    "fit2": make_draws(-1.3, [0.8, 0.2]),
    "fit3": make_draws(-1.7, [0.7, 0.3]),
}

# extreme log_lik points to test that models can handle them
MODEL_DRAWS_EXTREME = copy.copy(MODEL_DRAWS)
MODEL_DRAWS_EXTREME["fit1"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)
MODEL_DRAWS_EXTREME["fit2"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)
MODEL_DRAWS_EXTREME["fit3"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)

DISCRETE_COVARIATES = {
    "dummy": ["group1"] * (BERN_DATA["N"] // 2) + ["group2"] * (BERN_DATA["N"] // 2)
}

CONTINUOUS_COVARIATES = {"metric": RNG.normal(size=BERN_DATA["N"])}

CONTINUOUS_COVARIATES_ZERO = {"metric_zero": np.zeros(BERN_DATA["N"])}


@lru_cache
def hierarchical_bayes_stacking():
    return HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        discrete_covariates=DISCRETE_COVARIATES,
        seed=SEED,
    ).fit()


@lru_cache
def hierarchical_bayes_stacking_pooling():
    return HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        discrete_covariates=DISCRETE_COVARIATES,
        partial_pooling=True,
        seed=SEED,
    ).fit()


@lru_cache
def fit_models():
    mle_stacking = MleStacking(model_draws=MODEL_DRAWS).fit()
    bayes_stacking = BayesStacking(model_draws=MODEL_DRAWS, seed=SEED).fit()
    hier_bayes_stacking = hierarchical_bayes_stacking()
    pseudo_bma = PseudoBma(
        model_draws=MODEL_DRAWS,
        bootstrap=False,
        seed=SEED,
    ).fit()
    pseudo_bma_plus = PseudoBma(
        model_draws=MODEL_DRAWS,
        n_boots=1000,
        seed=SEED,
    ).fit()

    return (
        mle_stacking,
        bayes_stacking,
        hier_bayes_stacking,
        pseudo_bma,
        pseudo_bma_plus,
    )


def test_simple_blend_valid_predictions():
    w = {k: 1 / len(MODEL_DRAWS) for k in MODEL_DRAWS}
    model = SimpleBlend(model_draws=MODEL_DRAWS, weights=w)
    blend = model.predict()

    target_ll = np.array([draws.log_lik for draws in MODEL_DRAWS.values()])
    assert isinstance(model, SimpleBlend)
    assert model.weights
    assert np.isclose(blend.log_lik.mean(), target_ll.mean())


def test_simple_blend_catches_errors():
    with pytest.raises(ValueError):
        w = {k: [[[1 / len(MODEL_DRAWS)]]] for k in MODEL_DRAWS}
        SimpleBlend(model_draws=MODEL_DRAWS, weights=w)
    with pytest.raises(ValueError):
        w = {k: 0.5 for k in range(4)}
        SimpleBlend(model_draws=MODEL_DRAWS, weights=w)


def test_model_weights_valid():
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models()
    )

    assert sum(mle_stacking.weights.values())
    assert sum(bayes_stacking.weights.values())
    assert all(sum(hier_bayes_stacking.weights.values())[0])
    assert sum(pseudo_bma.weights.values())
    assert sum(pseudo_bma_plus.weights.values())


def test_model_blending_valid():
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models()
    )

    assert isinstance(mle_stacking._blend(), Draws)
    assert isinstance(bayes_stacking._blend(), Draws)
    assert isinstance(hier_bayes_stacking._blend(), Draws)
    assert isinstance(pseudo_bma._blend(), Draws)
    assert isinstance(pseudo_bma_plus._blend(), Draws)


def test_model_predictions_valid():
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models()
    )

    assert isinstance(mle_stacking.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(bayes_stacking.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(
        hier_bayes_stacking.predict(return_weights=True, seed=SEED), tuple
    )
    assert isinstance(pseudo_bma.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(pseudo_bma_plus.predict(return_weights=True, seed=SEED), tuple)


def test_equal_diagnostics_equal_weights():
    model_draws = dict(model1=MODEL_DRAWS["fit1"], model2=MODEL_DRAWS["fit1"])
    stacking = MleStacking(model_draws=model_draws).fit()
    assert all([w == 0.5 for w in stacking.weights.values()])


def test_bayes_stacking_weight_extreme_elpd():
    mle_stacking = MleStacking(model_draws=MODEL_DRAWS_EXTREME).fit()
    bayes_stacking = BayesStacking(model_draws=MODEL_DRAWS_EXTREME, seed=SEED).fit()
    # MLE optimization will fail to converge, Bayes will succeed
    assert not mle_stacking.model_info.success
    assert bayes_stacking.model_info.summary()["R_hat"].max() < 1.01


def test_bayes_stacking_weight_arrays():
    bayes_stacking = BayesStacking(model_draws=MODEL_DRAWS, seed=SEED).fit()
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
            model_draws=MODEL_DRAWS,
        )


def test_hier_bayes_stacking_predict_with_new():
    hier_bayes_stacking = hierarchical_bayes_stacking()

    # predicting with new data will fail if new covariates given without
    # new draws, and vice-versa. This is to catch issues where someone
    # tries to predict by, e.g., only passing new covariates, where the
    # model would then try to use those covariates to blend draws used
    # to fit the averaging/stacking model.
    with pytest.raises(ValueError, match="Either `model_draws`"):
        hier_bayes_stacking.predict(discrete_covariates=DISCRETE_COVARIATES)

    with pytest.raises(ValueError, match="Either `model_draws`"):
        hier_bayes_stacking.predict(model_draws=MODEL_DRAWS)

    assert hier_bayes_stacking.predict(
        model_draws=MODEL_DRAWS, discrete_covariates=DISCRETE_COVARIATES, seed=SEED
    )


def test_hier_bayes_stacking_only_continuous_covariates():
    hier_bayes_stacking = HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        continuous_covariates=CONTINUOUS_COVARIATES,
        cmdstan_control=CFG,
        seed=SEED,
    ).fit()

    assert hier_bayes_stacking.predict(seed=SEED)


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
        hier_bayes_stacking._predict_weights(discrete_covariates=new_level)

    # covariate name is the same, but there is a missing level
    # should generate predictions as normal without failing
    missing_level = {"dummy": ["group1"] * 4}
    hier_bayes_stacking._predict_weights(discrete_covariates=missing_level)


def test_hier_bayes_stacking_pooling():
    hier_bayes_stacking_pooled = hierarchical_bayes_stacking_pooling()

    hyperprior_pars = ["mu_global"]
    prior_pars = ["mu_disc", "sigma_disc"]

    n_models = len(MODEL_DRAWS)
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
        hier_bayes_stacking._predict_weights(discrete_covariates=one_new_level)

    two_new_levels = {"dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2}
    with pytest.warns(UserWarning):
        hier_bayes_stacking._predict_weights(discrete_covariates=two_new_levels)

    three_new_levels = {
        "dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2 + ["group97"] * 2
    }
    with pytest.warns(UserWarning):
        hier_bayes_stacking._predict_weights(discrete_covariates=three_new_levels)


def test_hier_bayes_stacking_continuous_covariates_transform():
    transforms = ["identity", "standardize", "relu"]
    stacks = {}
    for transform in transforms:
        stacks[transform] = HierarchicalBayesStacking(
            model_draws=MODEL_DRAWS,
            continuous_covariates=CONTINUOUS_COVARIATES,
            continuous_covariates_transform=transform,
            cmdstan_control=CFG,
            seed=SEED,
        ).fit()

    assert all(k == v.continuous_covariates_transform for k, v in stacks.items())
    assert len(stacks["relu"].coefficients["beta_cont"][0]) == 2

    with pytest.raises(ValueError, match="logit not found"):
        HierarchicalBayesStacking(
            model_draws=MODEL_DRAWS,
            continuous_covariates=CONTINUOUS_COVARIATES,
            continuous_covariates_transform="logit",
            cmdstan_control=CFG,
        ).fit()

    # error if trying to standardize variable with 0 SD/variance
    with pytest.raises(ValueError, match="cannot be standardized"):
        HierarchicalBayesStacking(
            model_draws=MODEL_DRAWS,
            continuous_covariates=CONTINUOUS_COVARIATES_ZERO,
            continuous_covariates_transform="standardize",
            cmdstan_control=CFG,
        ).fit()


def test_hier_bayes_stacking_weight_predictions():
    hier_bayes_stacking = hierarchical_bayes_stacking()
    weight_predictions_old_data = hier_bayes_stacking._predict_weights(
        discrete_covariates=DISCRETE_COVARIATES
    )
    new_covariates = {"dummy": ["group1"] * 2 + ["group2"] * 2}
    weight_predictions_new_data = hier_bayes_stacking._predict_weights(
        discrete_covariates=new_covariates,
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
    new_draws = MODEL_DRAWS

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
    blended_draws, predicted_weights = hier_bayes_stacking.predict(
        model_draws=new_draws,
        discrete_covariates=new_levels,
        return_weights=True,
        seed=SEED,
    )

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
    assert isinstance(blended_draws, Draws)


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

    bayes_stacking = BayesStacking(
        model_draws=MODEL_DRAWS, cmdstan_control=CFG, seed=SEED
    ).fit()
    bayes_stacking_priors = BayesStacking(
        model_draws=MODEL_DRAWS,
        cmdstan_control=CFG,
        priors={"w_prior": [2, 2, 2]},
        seed=SEED,
    ).fit()
    hier_bayes_stacking = HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        cmdstan_control=CFG,
        discrete_covariates=DISCRETE_COVARIATES,
        seed=SEED,
    ).fit()
    hier_bayes_stacking_priors = HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        cmdstan_control=CFG,
        discrete_covariates=DISCRETE_COVARIATES,
        priors=hier_priors,
        seed=SEED,
    ).fit()

    assert bayes_stacking.priors == {"w_prior": [1, 1, 1]}
    assert bayes_stacking_priors.priors == {"w_prior": [2, 2, 2]}
    assert hier_bayes_stacking.priors == HierarchicalBayesStacking.DEFAULT_PRIORS
    assert hier_bayes_stacking_priors.priors == hier_priors

    with pytest.raises(ValueError):
        BayesStacking(model_draws=MODEL_DRAWS, priors={"w_prior": [1]})

    with pytest.raises(ValueError):
        HierarchicalBayesStacking(
            model_draws=MODEL_DRAWS,
            discrete_covariates=DISCRETE_COVARIATES,
            priors=hier_priors | {"x": 1},
        )


def test_bayes_stacking_adaptive_prior_structure():
    base = hierarchical_bayes_stacking()
    adaptive = HierarchicalBayesStacking(
        model_draws=MODEL_DRAWS,
        discrete_covariates=DISCRETE_COVARIATES,
        adaptive=True,
        cmdstan_control=CFG,
        seed=SEED,
    ).fit()

    _lambda = adaptive.model_info.stan_variable("lambda")
    assert adaptive.adaptive
    assert not base.adaptive
    assert "lambda" in adaptive.model_info.stan_variables().keys()
    assert "lambda" not in base.model_info.stan_variables().keys()
    assert np.isclose(_lambda.mean(), 0.3, rtol=1e-1)


def test_blend_3d_variables():
    model_draws = {
        "fit1": make_draws(-1, [0.9, 0.1], shape=(100, 10, 10)),
        "fit2": make_draws(-1.3, [0.8, 0.2], shape=(100, 10, 10)),
        "fit3": make_draws(-1.7, [0.7, 0.3], shape=(100, 10, 10)),
    }
    blend = MleStacking(model_draws=model_draws).fit().predict(seed=SEED)

    assert isinstance(blend, Draws)
    assert all(blend.shape == draws.shape for draws in model_draws.values())


def test_models_from_cmdstanpy():
    model_fits = dict(fit1=FIT, fit2=FIT)
    assert MleStacking.from_cmdstanpy(model_fits)


def test_models_io_arviz():
    idata = {
        "fit1": az.from_cmdstanpy(FIT),
        "fit2": az.from_cmdstanpy(FIT),
    }
    stack = MleStacking.from_arviz(idata)
    stack.fit()
    blend = stack.predict()
    arviz_blend = blend.to_arviz(dims=(4, 1000, 10))
    assert isinstance(arviz_blend, az.InferenceData)
    assert np.all(np.vstack(arviz_blend.log_likelihood.log_lik.values) == blend.log_lik)
    assert np.all(
        np.vstack(arviz_blend.posterior_predictive.post_pred.values) == blend.post_pred
    )
    with pytest.warns(UserWarning, match=r"More chains \(4000\) than draws \(10\)."):
        blend.to_arviz()


def test_models_from_lpd():
    # Generate some fake LPDs.
    # note, we just take the mean here,
    # not the logmeanexp
    lpds = {name: fit.log_lik.mean(axis=0) for name, fit in MODEL_DRAWS.items()}
    post_preds = {name: fit.post_pred for name, fit in MODEL_DRAWS.items()}
    assert MleStacking.from_lpd(lpds, post_preds)


def test_model_from_lpd_3d():
    model_draws = {
        "fit1": make_draws(-1, [0.9, 0.1], shape=(100, 10, 10)),
        "fit2": make_draws(-1.3, [0.8, 0.2], shape=(100, 10, 10)),
        "fit3": make_draws(-1.7, [0.7, 0.3], shape=(100, 10, 10)),
    }
    lpds = {name: fit.log_lik.mean(axis=0) for name, fit in model_draws.items()}
    post_preds = {name: fit.post_pred for name, fit in model_draws.items()}
    assert MleStacking.from_lpd(lpds, post_preds)


def test_seed():
    blend1 = MleStacking(model_draws=MODEL_DRAWS).fit().predict(seed=1)
    blend2 = MleStacking(model_draws=MODEL_DRAWS).fit().predict(seed=2)
    blend3 = MleStacking(model_draws=MODEL_DRAWS).fit().predict(seed=1)

    assert np.array_equal(blend1.log_lik, blend3.log_lik)
    assert not np.array_equal(blend1.log_lik, blend2.log_lik)
