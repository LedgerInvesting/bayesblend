import pytest
import json
import numpy as np
from cmdstanpy import CmdStanModel

from bayesblend import (
    BayesStacking,
    HierarchicalBayesStacking,
    MleStacking,
    PseudoBma,
    Draws,
)

SEED = 1234

RNG = np.random.default_rng(SEED)

STAN_FILE = "test/stan_files/bernoulli_ppc.stan"
DATA_FILE = "test/stan_data/bernoulli_data.json"

MODEL = CmdStanModel(stan_file=STAN_FILE)

CFG = {"chains": 4, "parallel_chains": 4}

SEED = 1234

with open(DATA_FILE, "r") as f:
    BERN_DATA = json.load(f)


RNG = np.random.RandomState(SEED)


@pytest.fixture
def fit():
    return MODEL.sample(data=BERN_DATA, **CFG)


@pytest.fixture
def discrete_covariates():
    return {
        "dummy": ["group1"] * (BERN_DATA["N"] // 2) + ["group2"] * (BERN_DATA["N"] // 2)
    }


@pytest.fixture
def continuous_covariates():
    return {"metric": RNG.normal(size=BERN_DATA["N"])}


@pytest.fixture
def continuous_covariates_zero():
    return {"metric_zero": np.zeros(BERN_DATA["N"])}


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


@pytest.fixture
def model_draws():
    return {
        "fit1": make_draws(-1, [0.9, 0.1]),
        "fit2": make_draws(-1.3, [0.8, 0.2]),
        "fit3": make_draws(-1.7, [0.7, 0.3]),
    }


@pytest.fixture
def model_draws_3d():
    return {
        "fit1": make_draws(-1, [0.9, 0.1], shape=(100, 10, 10)),
        "fit2": make_draws(-1.3, [0.8, 0.2], shape=(100, 10, 10)),
        "fit3": make_draws(-1.7, [0.7, 0.3], shape=(100, 10, 10)),
    }


@pytest.fixture
def model_draws_extreme(model_draws):
    model_draws["fit1"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)
    model_draws["fit2"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)
    model_draws["fit3"].log_lik[:, 0] = RNG.normal(-1e5, 0.1, 1000)
    return model_draws


@pytest.fixture
def hierarchical_bayes_stacking(model_draws, discrete_covariates):
    return HierarchicalBayesStacking(
        model_draws=model_draws,
        discrete_covariates=discrete_covariates,
        seed=SEED,
    ).fit()


@pytest.fixture
def hierarchical_bayes_stacking_pooling(model_draws, discrete_covariates):
    return HierarchicalBayesStacking(
        model_draws=model_draws,
        discrete_covariates=discrete_covariates,
        partial_pooling=True,
        seed=SEED,
    ).fit()


@pytest.fixture
def hierarchical_bayes_stacking_pooling_two_discrete_covariates(
    model_draws, discrete_covariates
):
    new_covariate = {}
    new_covariate["dummy2"] = discrete_covariates["dummy"]
    discrete_covariates = discrete_covariates | new_covariate
    return HierarchicalBayesStacking(
        model_draws=model_draws,
        discrete_covariates=discrete_covariates,
        partial_pooling=True,
        seed=SEED,
    ).fit()


@pytest.fixture
def fit_models(model_draws, hierarchical_bayes_stacking):
    mle_stacking = MleStacking(model_draws=model_draws, seed=SEED).fit()
    bayes_stacking = BayesStacking(model_draws=model_draws, seed=SEED).fit()
    pseudo_bma = PseudoBma(
        model_draws=model_draws,
        bootstrap=False,
        seed=SEED,
    ).fit()
    pseudo_bma_plus = PseudoBma(
        model_draws=model_draws,
        n_boots=1000,
        seed=SEED,
    ).fit()

    return (
        mle_stacking,
        bayes_stacking,
        hierarchical_bayes_stacking,
        pseudo_bma,
        pseudo_bma_plus,
    )
