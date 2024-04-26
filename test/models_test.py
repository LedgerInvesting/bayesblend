import pytest
import arviz as az
import numpy as np
from bayesblend.models import _make_dummy_vars

from bayesblend import (
    BayesStacking,
    HierarchicalBayesStacking,
    MleStacking,
    SimpleBlend,
    Draws,
)

from .fixtures import SEED, CFG, BERN_DATA


def test_simple_blend_valid_predictions(model_draws):
    w = {k: 1 / len(model_draws) for k in model_draws}
    model = SimpleBlend(model_draws=model_draws, weights=w)
    blend = model.predict()

    target_ll = np.array([draws.log_lik for draws in model_draws.values()])
    assert isinstance(model, SimpleBlend)
    assert model.weights
    assert np.isclose(blend.log_lik.mean(), target_ll.mean(), rtol=1e-1)


def test_simple_blend_catches_errors(model_draws):
    with pytest.raises(ValueError):
        w = {k: [[[1 / len(model_draws)]]] for k in model_draws}
        SimpleBlend(model_draws=model_draws, weights=w)
    with pytest.raises(ValueError):
        w = {k: 0.5 for k in range(4)}
        SimpleBlend(model_draws=model_draws, weights=w)


def test_model_weights_valid(fit_models):
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models
    )

    assert sum(mle_stacking.weights.values())
    assert sum(bayes_stacking.weights.values())
    assert all(sum(hier_bayes_stacking.weights.values())[0])
    assert sum(pseudo_bma.weights.values())
    assert sum(pseudo_bma_plus.weights.values())


def test_model_blending_valid(fit_models):
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models
    )

    assert isinstance(mle_stacking._blend(), Draws)
    assert isinstance(bayes_stacking._blend(), Draws)
    assert isinstance(hier_bayes_stacking._blend(), Draws)
    assert isinstance(pseudo_bma._blend(), Draws)
    assert isinstance(pseudo_bma_plus._blend(), Draws)


def test_model_blending_reproducible(fit_models):
    models = fit_models

    for model in models:
        assert model.fit().weights == model.fit().weights
        assert model.predict().lpd.sum() == model.predict().lpd.sum()


def test_model_predictions_valid(fit_models):
    mle_stacking, bayes_stacking, hier_bayes_stacking, pseudo_bma, pseudo_bma_plus = (
        fit_models
    )

    assert isinstance(mle_stacking.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(bayes_stacking.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(
        hier_bayes_stacking.predict(return_weights=True, seed=SEED), tuple
    )
    assert isinstance(pseudo_bma.predict(return_weights=True, seed=SEED), tuple)
    assert isinstance(pseudo_bma_plus.predict(return_weights=True, seed=SEED), tuple)


def test_equal_diagnostics_equal_weights(model_draws):
    model_draws = dict(model1=model_draws["fit1"], model2=model_draws["fit1"])
    stacking = MleStacking(model_draws=model_draws).fit()
    assert all([w == 0.5 for w in stacking.weights.values()])


def test_bayes_stacking_weight_extreme_elpd(model_draws_extreme):
    mle_stacking = MleStacking(model_draws=model_draws_extreme).fit()
    bayes_stacking = BayesStacking(model_draws=model_draws_extreme, seed=SEED).fit()
    # MLE optimization will fail to converge, Bayes will succeed
    assert not mle_stacking.model_info.success
    assert bayes_stacking.model_info.summary()["R_hat"].max() < 1.01


def test_bayes_stacking_weight_arrays(model_draws, hierarchical_bayes_stacking):
    bayes_stacking = BayesStacking(model_draws=model_draws, seed=SEED).fit()
    # internal weights are full posteriors, and should always have shape[0]>1
    # hierarchical stacking should also have weights for each datapoint, or shape[1]==N
    assert all(v.shape[0] > 1 for v in bayes_stacking._weights.values())
    assert all(v.shape[1] == 1 for v in bayes_stacking._weights.values())
    assert all(v.shape[0] > 1 for v in hierarchical_bayes_stacking._weights.values())
    assert all(
        v.shape[1] == BERN_DATA["N"]
        for v in hierarchical_bayes_stacking._weights.values()
    )

    # external/user-readable weights should be posterior means, reducing to shape[0]==1
    # Otherwise, dimenions should be same as above
    assert all(v.shape[0] == 1 for v in bayes_stacking.weights.values())
    assert all(v.shape[1] == 1 for v in bayes_stacking.weights.values())
    assert all(v.shape[0] == 1 for v in hierarchical_bayes_stacking.weights.values())
    assert all(
        v.shape[1] == BERN_DATA["N"]
        for v in hierarchical_bayes_stacking.weights.values()
    )


def test_hier_bayes_stacking_no_covariates_fails(model_draws):
    with pytest.raises(ValueError):
        HierarchicalBayesStacking(
            model_draws=model_draws,
        )


def test_hier_bayes_stacking_predict_with_new(
    model_draws, hierarchical_bayes_stacking, discrete_covariates
):
    # predicting with new data will fail if new covariates given without
    # new draws, and vice-versa. This is to catch issues where someone
    # tries to predict by, e.g., only passing new covariates, where the
    # model would then try to use those covariates to blend draws used
    # to fit the averaging/stacking model.
    with pytest.raises(ValueError, match="Either `model_draws`"):
        hierarchical_bayes_stacking.predict(discrete_covariates=discrete_covariates)

    with pytest.raises(ValueError, match="Either `model_draws`"):
        hierarchical_bayes_stacking.predict(model_draws=model_draws)

    assert hierarchical_bayes_stacking.predict(
        model_draws=model_draws, discrete_covariates=discrete_covariates, seed=SEED
    )


def test_hier_bayes_stacking_only_continuous_covariates(
    model_draws, continuous_covariates
):
    hier_bayes_stacking = HierarchicalBayesStacking(
        model_draws=model_draws,
        continuous_covariates=continuous_covariates,
        cmdstan_control=CFG,
        seed=SEED,
    ).fit()

    assert hier_bayes_stacking.predict(seed=SEED)


def test_hier_bayes_stacking_predict_different_covariates_fails(
    hierarchical_bayes_stacking,
):
    different_covariates = {"fail": ["group1"] * 2 + ["group2"] * 2}
    with pytest.raises(ValueError):
        hierarchical_bayes_stacking.predict(discrete_covariates=different_covariates)


def test_hier_bayes_stacking_predict_different_covariate_levels(
    hierarchical_bayes_stacking,
):
    # covariate name is the same, but there is a new level
    new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    with pytest.raises(ValueError):
        hierarchical_bayes_stacking._predict_weights(discrete_covariates=new_level)

    # covariate name is the same, but there is a missing level
    # should generate predictions as normal without failing
    missing_level = {"dummy": ["group1"] * 4}
    hierarchical_bayes_stacking._predict_weights(discrete_covariates=missing_level)


def test_hier_bayes_stacking_pooling(model_draws, hierarchical_bayes_stacking_pooling):
    hyperprior_pars = ["mu_global"]
    prior_pars = ["mu_disc", "sigma_disc"]

    n_models = len(model_draws)
    model_coefs = hierarchical_bayes_stacking_pooling.coefficients

    assert all(
        par in hierarchical_bayes_stacking_pooling.coefficients
        for par in hyperprior_pars + prior_pars
    )
    # check shape of model hyperprior parameters
    assert all(model_coefs[par].shape == (1, 1) for par in hyperprior_pars)
    # check shape of model prior parameters
    assert all(model_coefs[par].shape == (1, n_models - 1) for par in prior_pars)


def test_make_dummy_vars_new_levels(hierarchical_bayes_stacking_pooling):
    discrete_covariate_info = hierarchical_bayes_stacking_pooling.covariate_info

    one_new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    dummies = _make_dummy_vars(one_new_level, discrete_covariate_info)
    assert list(dummies.keys()) == ["dummy_group2", "dummy_group99"]
    assert not all(dummies["dummy_group2"])
    assert dummies["dummy_group99"] == [0, 0, 1, 1]

    two_new_levels = {"dummy": ["group1"] * 2 + ["group0"] * 2 + ["group99"] * 2}
    dummies = _make_dummy_vars(two_new_levels, discrete_covariate_info)
    assert list(dummies.keys()) == ["dummy_group2", "dummy_group0", "dummy_group99"]
    assert not all(dummies["dummy_group2"])
    assert dummies["dummy_group0"] == [0, 0, 1, 1, 0, 0]
    assert dummies["dummy_group99"] == [0, 0, 0, 0, 1, 1]

    train = {"dummy": ["group1"] * 3 + ["group10"] * 2}
    test = {"dummy": ["group1"] * 1 + ["group01"] * 2}
    dummies = _make_dummy_vars(train)
    test_dummies = _make_dummy_vars(
        test,
        {"dummy": ["group1", "group10"]},
    )

    assert dummies["dummy_group10"] == [0, 0, 0, 1, 1]
    assert list(test_dummies.keys()) == ["dummy_group10", "dummy_group01"]
    assert test_dummies["dummy_group10"] == [0, 0, 0]
    assert test_dummies["dummy_group01"] == [0, 1, 1]


def test_make_dummy_vars_new_levels_two_covariates(
    hierarchical_bayes_stacking_pooling_two_discrete_covariates,
):
    discrete_covariate_info = (
        hierarchical_bayes_stacking_pooling_two_discrete_covariates.covariate_info
    )

    one_new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    one_new_level["dummy2"] = one_new_level["dummy"]
    dummies = _make_dummy_vars(one_new_level, discrete_covariate_info)
    assert list(dummies.keys()) == [
        "dummy_group2",
        "dummy2_group2",
        "dummy_group99",
        "dummy2_group99",
    ]
    assert not all(dummies["dummy_group2"]) and not all(dummies["dummy2_group2"])
    assert dummies["dummy_group99"] == [0, 0, 1, 1]
    assert dummies["dummy2_group99"] == [0, 0, 1, 1]


def test_hier_bayes_stacking_predict_different_covariate_levels_pooling(
    hierarchical_bayes_stacking_pooling,
):
    # covariate name is the same, but there is a new level
    one_new_level = {"dummy": ["group1"] * 2 + ["group99"] * 2}
    with pytest.warns(UserWarning):
        hierarchical_bayes_stacking_pooling._predict_weights(
            discrete_covariates=one_new_level
        )

    one_new_level_prev_index = {"dummy": ["group0"] * 2 + ["group1"] * 2}
    with pytest.warns(UserWarning):
        hierarchical_bayes_stacking_pooling._predict_weights(
            discrete_covariates=one_new_level_prev_index
        )

    two_new_levels = {"dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2}
    with pytest.warns(UserWarning):
        hierarchical_bayes_stacking_pooling._predict_weights(
            discrete_covariates=two_new_levels
        )

    three_new_levels = {
        "dummy": ["group1"] * 2 + ["group99"] * 2 + ["group98"] * 2 + ["group97"] * 2
    }
    with pytest.warns(UserWarning):
        hierarchical_bayes_stacking_pooling._predict_weights(
            discrete_covariates=three_new_levels
        )


def test_hier_bayes_stacking_continuous_covariates_transform(
    model_draws, continuous_covariates, continuous_covariates_zero
):
    transforms = ["identity", "standardize", "relu"]
    stacks = {}
    for transform in transforms:
        stacks[transform] = HierarchicalBayesStacking(
            model_draws=model_draws,
            continuous_covariates=continuous_covariates,
            continuous_covariates_transform=transform,
            cmdstan_control=CFG,
            seed=SEED,
        ).fit()

    assert all(k == v.continuous_covariates_transform for k, v in stacks.items())
    assert len(stacks["relu"].coefficients["beta_cont"][0]) == 2

    with pytest.raises(ValueError, match="logit not found"):
        HierarchicalBayesStacking(
            model_draws=model_draws,
            continuous_covariates=continuous_covariates,
            continuous_covariates_transform="logit",
            cmdstan_control=CFG,
        ).fit()

    # error if trying to standardize variable with 0 SD/variance
    with pytest.raises(ValueError, match="cannot be standardized"):
        HierarchicalBayesStacking(
            model_draws=model_draws,
            continuous_covariates=continuous_covariates_zero,
            continuous_covariates_transform="standardize",
            cmdstan_control=CFG,
        ).fit()


def test_hier_bayes_stacking_weight_predictions(
    hierarchical_bayes_stacking, discrete_covariates
):
    weight_predictions_old_data = hierarchical_bayes_stacking._predict_weights(
        discrete_covariates=discrete_covariates
    )
    new_covariates = {"dummy": ["group1"] * 2 + ["group2"] * 2}
    weight_predictions_new_data = hierarchical_bayes_stacking._predict_weights(
        discrete_covariates=new_covariates,
    )

    # predictions from passing in same data should be approx equal to estimated weights
    assert all(
        np.allclose(true, pred, atol=0.001)
        for true, pred in zip(
            hierarchical_bayes_stacking.weights.values(),
            weight_predictions_old_data.values(),
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


def test_hier_bayes_stacking_predictions_new_level_dummy_codes(
    model_draws, hierarchical_bayes_stacking_pooling
):
    # covariate name is the same, but there are new levels
    new_levels = {"dummy": ["group2"] * 5 + ["group10"] * 5}
    new_draws = model_draws

    # set the group-level covariate coefficent to all 0s to force
    # new level coefficent to be exactly 0. The estimated weights
    # will then be the same as the predicted weights. This ensures
    # that the dummy coding for new levels is working as intended
    hierarchical_bayes_stacking_pooling._coefficients["mu_disc"] = np.zeros_like(
        hierarchical_bayes_stacking_pooling._coefficients["mu_disc"]
    )
    hierarchical_bayes_stacking_pooling._coefficients["sigma_disc"] = np.zeros_like(
        hierarchical_bayes_stacking_pooling._coefficients["sigma_disc"]
    )
    blended_draws, predicted_weights = hierarchical_bayes_stacking_pooling.predict(
        model_draws=new_draws,
        discrete_covariates=new_levels,
        return_weights=True,
        seed=SEED,
    )

    # predicitons where dummy=group10 should be the same as when dummy=group1,
    # and predictions for dummy=group2 should remain the same as before. Because
    # original weights are ordered per discrete_covariates, predictions should
    # match reversed original weights
    assert all(
        np.allclose(true[:, ::-1], pred, atol=0.001)
        for true, pred in zip(
            hierarchical_bayes_stacking_pooling.weights.values(),
            predicted_weights.values(),
        )
    )
    assert isinstance(blended_draws, Draws)


def test_bayes_stacking_generation_with_priors(model_draws, discrete_covariates):
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
        model_draws=model_draws, cmdstan_control=CFG, seed=SEED
    ).fit()
    bayes_stacking_priors = BayesStacking(
        model_draws=model_draws,
        cmdstan_control=CFG,
        priors={"w_prior": [2, 2, 2]},
        seed=SEED,
    ).fit()
    hier_bayes_stacking = HierarchicalBayesStacking(
        model_draws=model_draws,
        cmdstan_control=CFG,
        discrete_covariates=discrete_covariates,
        seed=SEED,
    ).fit()
    hier_bayes_stacking_priors = HierarchicalBayesStacking(
        model_draws=model_draws,
        cmdstan_control=CFG,
        discrete_covariates=discrete_covariates,
        priors=hier_priors,
        seed=SEED,
    ).fit()

    assert bayes_stacking.priors == {"w_prior": [1, 1, 1]}
    assert bayes_stacking_priors.priors == {"w_prior": [2, 2, 2]}
    assert hier_bayes_stacking.priors == HierarchicalBayesStacking.DEFAULT_PRIORS
    assert hier_bayes_stacking_priors.priors == hier_priors

    with pytest.raises(ValueError):
        BayesStacking(model_draws=model_draws, priors={"w_prior": [1]})

    with pytest.raises(ValueError):
        HierarchicalBayesStacking(
            model_draws=model_draws,
            discrete_covariates=discrete_covariates,
            priors=hier_priors | {"x": 1},
        )


def test_bayes_stacking_adaptive_prior_structure(
    model_draws, discrete_covariates, hierarchical_bayes_stacking
):
    base = hierarchical_bayes_stacking
    adaptive = HierarchicalBayesStacking(
        model_draws=model_draws,
        discrete_covariates=discrete_covariates,
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


def test_blend_3d_variables(model_draws_3d):
    blend = MleStacking(model_draws=model_draws_3d).fit().predict(seed=SEED)

    assert isinstance(blend, Draws)
    assert all(blend.shape == draws.shape for draws in model_draws_3d.values())


def test_models_from_cmdstanpy(fit):
    model_fits = dict(fit1=fit, fit2=fit)
    assert MleStacking.from_cmdstanpy(model_fits)


def test_models_io_arviz(fit):
    idata = {
        "fit1": az.from_cmdstanpy(fit),
        "fit2": az.from_cmdstanpy(fit),
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


def test_models_from_lpd(model_draws):
    # Generate some fake LPDs.
    # note, we just take the mean here,
    # not the logmeanexp
    lpds = {name: fit.log_lik.mean(axis=0) for name, fit in model_draws.items()}
    post_preds = {name: fit.post_pred for name, fit in model_draws.items()}
    assert MleStacking.from_lpd(lpds, post_preds)


def test_model_from_lpd_3d(model_draws_3d):
    lpds = {name: fit.log_lik.mean(axis=0) for name, fit in model_draws_3d.items()}
    post_preds = {name: fit.post_pred for name, fit in model_draws_3d.items()}
    assert MleStacking.from_lpd(lpds, post_preds)
