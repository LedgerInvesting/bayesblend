import numpy as np
from scipy.stats import gaussian_kde
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
X = np.hstack([X, np.prod(X, axis=1).reshape((N, 1))])
beta = np.array([1.5, 0.2, 0.5])

y = rng.normal(alpha + X @ beta, sigma)

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
        simplex[K] w;
    }

    model {
        alpha ~ normal(0, 10);
        beta ~ normal(0, 10);
        sigma ~ normal(0, 10);

        for(i in 1:N) {
            vector[K] lps = [
                normal_lpdf(y[i] | alpha + beta[1] * X[i,1], sigma),
                normal_lpdf(y[i] | alpha + beta[2] * X[i,2], sigma),
                normal_lpdf(y[i] | alpha + X[i] * beta, sigma)
            ]';
            target += log_sum_exp(log(w) + lps);
        }
    }

    generated quantities {
        vector[N] post_pred;

        for(i in 1:N) {
            vector[K] preds = [
                normal_rng(alpha + beta[1] * X[i,1], sigma),
                normal_rng(alpha + beta[2] * X[i,2], sigma),
                normal_rng(alpha + X[i] * beta, sigma)
            ]';
            int mix_idx = categorical_rng(w);
            post_pred[i] = preds[mix_idx];
        }
    }
"""

with open("mixture.stan", "w") as stan_file:
    stan_file.write(mixture_string)

mixture = csp.CmdStanModel(stan_file="mixture.stan")

fit_mixture = mixture.sample(
    data={"N": N, "P": P, "K": K, "X": X[:,:2], "y": y},
    seed=SEED
)

mixture_weights = fit_mixture.stan_variables()["w"]

def density(x):
    limits = x.min(), x.max()
    grid = np.linspace(*limits, 1000)
    return grid, gaussian_kde(x)(grid)

fig, ax = plt.subplots(1, 2, figsize=(8, 5))
fills = ["indianred", "steelblue", "mediumseagreen"]
densities = list(zip(*[density(weight) for weight in mixture_weights.T]))
normalizer = np.sum(densities[1])
for k in range(K):
    ax[0].plot(densities[0][k], densities[1][k] / normalizer, color=fills[k], linewidth=0)
    ax[0].fill_between(
        densities[0][k],
        densities[1][k] / normalizer,
        [0]*len(densities[0][k]),
        color=fills[k],
        alpha=0.5,
        label=f"w({k + 1})",
    )
    ax[0].legend(frameon=False)
    ax[0].set_ylabel("density")
    ax[0].set_xlabel("weight")

ax[1].plot(*density(fit_mixture.stan_variables()["post_pred"].mean(axis=0)), linewidth=0)
ax[1].fill_between(
    *density(fit_mixture.stan_variables()["post_pred"].mean(axis=0)),
    [0]*len(densities[0][k]),
    color="skyblue",
    alpha=0.5,
    label=f"post pred",
)
ax[1].plot(*density(y), linewidth=0)
ax[1].fill_between(
    *density(y),
    [0]*len(densities[0][k]),
    color="grey",
    alpha=0.5,
    label=f"y",
)
ax[1].set_xlabel("y")
ax[1].legend(frameon=False, loc="upper right")

plt.savefig("docs/user-guide/scripts/figures/mixture-weights.png", dpi=120)
plt.close()

mixture_rmse = np.sqrt(
    np.mean(
        (fit_mixture.stan_variables()["post_pred"] - y)**2,
        axis=1
    ),
)

regression_string = """
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
    stan_file.write(regression_string)

regression = csp.CmdStanModel(stan_file="regression.stan")

regression_predictors = (
    X[:,0].reshape((N, 1)), 
    X[:,1].reshape((N, 1)), 
    X[:,:P].reshape((N, P)),
)

regression_fits = [
    regression.sample(
        data={"N": N, "P": x.shape[1], "X": x, "y": y}, seed=SEED
    )
    for x in regression_predictors
]

bma_plus = bb.PseudoBma.from_cmdstanpy(
        {f"fit{i}": fit for i, fit in enumerate(regression_fits)},
)
bma_plus.fit()
bma_plus_blend = bma_plus.predict()

bma_plus_rmse = np.sqrt(
    np.mean(
        (bma_plus_blend.post_pred - y)**2,
        axis=1
    ),
)

stack = bb.MleStacking.from_cmdstanpy(
        {f"fit{i}": fit for i, fit in enumerate(regression_fits)},
)
stack.fit()
stack_blend = stack.predict()

stack_rmse = np.sqrt(
    np.mean(
        (stack_blend.post_pred - y)**2,
        axis=1
    ),
)

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

scatter_kwargs = dict(facecolor="none", linewidth=2)
ax[0].scatter([1, 2, 3], mixture_weights.mean(axis=0), edgecolor="indianred", **scatter_kwargs, label="mixture model")
ax[0].scatter([1, 2, 3], list(bma_plus.weights.values()), edgecolor="steelblue", **scatter_kwargs, label="pseudo-BMA+")
ax[0].scatter([1, 2, 3], list(stack.weights.values()), edgecolor="mediumseagreen", **scatter_kwargs, label="stacking")
ax[0].set_xticks([1, 2, 3])
ax[0].set_xticklabels(["model 1", "model 2", "model 3"])
ax[0].legend(frameon=True)

ax[1].fill_between(
    *density(fit_mixture.stan_variables()["post_pred"].mean(axis=0)),
    [0]*len(densities[0][k]),
    color="indianred",
    alpha=0.3,
    label="mixture model",
)
ax[1].fill_between(
    *density(bma_plus_blend.post_pred.mean(axis=0)),
    [0]*len(densities[0][k]),
    color="steelblue",
    alpha=0.3,
    label="pseudo-BMA+",
)
ax[1].fill_between(
    *density(stack_blend.post_pred.mean(axis=0)),
    [0]*len(densities[0][k]),
    color="mediumseagreen",
    alpha=0.3,
    label="stacking"
)
ax[1].fill_between(
    *density(y),
    [0]*len(densities[0][k]),
    color="grey",
    alpha=0.3,
    label=f"y",
)
ax[1].set_xlabel("y")
ax[1].legend(frameon=False, loc="upper right")

ax[2].errorbar(
    ["mixture", "pseudo-BMA+", "stacking"],
    [
        mixture_rmse.mean(),
        bma_plus_rmse.mean(),
        stack_rmse.mean(),
    ],
    yerr = [
        mixture_rmse.std(),
        bma_plus_rmse.std(),
        stack_rmse.std(),
    ],
    fmt="o",
    color="black",
)

plt.savefig("docs/user-guide/scripts/figures/stacking-compare.png", dpi=120, bbox_inches="tight")
plt.close()
