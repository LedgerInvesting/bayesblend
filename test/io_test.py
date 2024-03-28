import json

import numpy as np
from cmdstanpy import CmdStanModel

from bayesblend.io import Draws

STAN_FILE = "test/stan_files/bernoulli_ppc.stan"
STAN_FILE_3D = "test/stan_files/bernoulli_ppc_3d.stan"
DATA_FILE = "test/stan_data/bernoulli_data.json"

with open(DATA_FILE, "r") as f:
    BERN_DATA = json.load(f)

MODEL = CmdStanModel(stan_file=STAN_FILE)


def test_draws_from_cmdstanpy():
    fit = MODEL.sample(data=BERN_DATA, chains=4, parallel_chains=4, seed=1234)
    assert Draws.from_cmdstanpy(fit)


def test_draws_from_cmdstanpy_3d_variables():
    MODEL_3D = CmdStanModel(stan_file=STAN_FILE_3D)
    y_data = np.stack([BERN_DATA["y"], BERN_DATA["y"]]).T
    fit = MODEL_3D.sample(
        data=BERN_DATA | {"D": 2, "y": y_data}, chains=4, parallel_chains=4, seed=1234
    )
    draws = Draws.from_cmdstanpy(fit)
    assert draws.log_lik_2d.ndim == 2
    assert draws.post_pred_2d.ndim == 2
