import numpy as np
import cmdstanpy as csp
import matplotlib.pyplot as plt

import bayesblend as bb

SEED = 1234

rng = np.random.default_rng(SEED)
N = 500
P = 2
K = 3
alpha = 0
sigma = 1
X = rng.normal(size=(N, P))
beta = np.array([1.5, 0.2])
w = [0.2, 0.2, 0.6]
W = np.array(w * N).reshape((N, K))

mus = np.array([
    alpha + beta[0] * X[:,0],
    alpha + beta[1] * X[:,1],
    alpha + X @ beta,
])

mixes = [rng.choice(range(K), p=w) for w in W]
y = np.array([
    rng.normal(mus[idx, i], sigma)
    for i, idx
    in enumerate(mixes)
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
        alpha ~ normal(0, 10);
        beta ~ normal(0, 10);
        sigma ~ normal(0, 10);

        y ~ normal(mu, sigma);
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] post_pred;

        for(i in 1:N) {
            log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
            post_pred[i] = normal_rng(mu[i], sigma);
        }
    }
"""

with open("regression.stan", "w") as stan_file:
    stan_file.write(model_string)

model = csp.CmdStanModel(stan_file="regression.stan")

mixture_string = """
    data {
        int<lower=0> N;
        int<lower=2> K;
        int<lower=1> P;
        matrix[N, P] X;
        vector[N] y;
    }

    parameters {
        real alpha;
        vector[P] beta;
        real<lower=0> sigma;
        vector[K - 1] w_raw;
    }

    transformed parameters {
        simplex[K] w;
        w[:K - 1] = inv_logit(w_raw);
        w[K] = 1 - sum(w[:K - 1]);
    }

    model {
        alpha ~ normal(0, 10);
        beta ~ normal(0, 10);
        sigma ~ normal(0, 10);
        w_raw ~ normal(0, 5);

        for(i in 1:N) {
            vector[K] targets = [
                normal_lpdf(y[i] | alpha + beta[1] * X[i,1], sigma),
                normal_lpdf(y[i] | alpha + beta[2] * X[i,2], sigma),
                normal_lpdf(y[i] | alpha + X[i,:] * beta, sigma)
            ]';
            target += log_mix(w, targets);
        }
    }

    generated quantities {
        vector[N] post_pred;

        for(i in 1:N) {
            vector[K] targets_pred = [
                normal_rng(alpha + beta[1] * X[i,1], sigma),
                normal_rng(alpha + beta[2] * X[i,2], sigma),
                normal_rng(alpha + X[i,:] * beta, sigma)
            ]';
            post_pred[i] = w' * targets_pred;
        }
    }

"""

with open("mixture.stan", "w") as stan_file:
    stan_file.write(mixture_string)

mixture = csp.CmdStanModel(stan_file="mixture.stan")

fit_mixture = mixture.sample(
    data={"N": N, "P": P, "K": K, "X": X, "y": y},
    seed=SEED
)

fits = [
    model.sample(data={"N": N, "P": x.shape[1], "X": x, "y": y}, seed=SEED)
    for x in (X[:,0].reshape((N, 1)), X[:,1].reshape((N, 1)), X)
]

stacking = bb.BayesStacking.from_cmdstanpy(
        {f"fit{i}": fit for i, fit in enumerate(fits)},
        seed=SEED,
)
stacking.fit()

weights = stacking._weights

R = 1
C = 3
fig, ax = plt.subplots(R, C, figsize=(8, 4))
ax[0].hist(weights["fit0"], bins=40, alpha=0.5)
ax[1].hist(weights["fit1"], bins=40, alpha=0.5)
ax[2].hist(weights["fit2"], bins=40, alpha=0.5)
ax[0].axvline(x=w[0], color="black", ls=":")
ax[1].axvline(x=w[1], color="black", ls=":")
ax[2].axvline(x=w[2], color="black", ls=":")
