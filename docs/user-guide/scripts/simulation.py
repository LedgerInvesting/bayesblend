import numpy as np
import cmdstanpy as csp

import bayesblend as bb

SEED = 1234

rng = np.random.default_rng(SEED)
N = 100
P = 2
K = 3
alpha = 0
sigma = 1
X = rng.normal(size=(N, P))
betas = np.array([1.5, 0.2])
W = np.array([0.15, 0.15, 0.7])

mus = np.array([
    alpha + betas[0] * X[:,0],
    alpha + betas[1] * X[:,1],
    alpha + X @ betas,
])

y = np.array([
    rng.normal(mus[idx - 1, i], sigma)
    for i, idx
    in enumerate(rng.choice(range(K), p=W, size=N))
])


model_string = """
    data {
        int<lower=0> N;
        int<lower=1> P;
        matrix[N, P] X;
        vector[N] y;
    }

    parameters {
        real alpha;
        vector[P] beta;
        real<lower=0> sigma;
    }

    transformed parameters {
        vector[N] mu = alpha + X * beta;
    }

    model {
        alpha ~ std_normal();
        beta ~ std_normal();
        sigma ~ std_normal();

        y ~ normal(mu, sigma);
    }

    generated quantities {
        vector[N] post_pred;
        vector[N] log_lik;

        for(i in 1:N) {
            log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
            post_pred[i] = normal_rng(mu[i], sigma);
        }
    }
"""

with open("regression.stan", "w") as stan_file:
    stan_file.write(model_string)

model = csp.CmdStanModel(stan_file="regression.stan")

fits = [
    model.sample(data={"N": N, "P": x.shape[1], "X": x, "y": y})
    for x in (X[:,0].reshape((N, 1)), X[:,1].reshape((N, 1)), X)
]

stacking = bb.MleStacking.from_cmdstanpy(
        {f"fit{i}": fit for i, fit in enumerate(fits)},
)
stacking.fit()
stacking.weights

