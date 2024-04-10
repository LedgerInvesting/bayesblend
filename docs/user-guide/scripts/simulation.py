# mypy: ignore-errors 

import arviz as az
import numpy as np
from scipy.stats import gaussian_kde
import cmdstanpy as csp
import matplotlib.pyplot as plt

import bayesblend as bb

SEED = 1234

def rmse(preds, y):
    se = (preds.mean(axis=0) - y)**2
    mse = se.mean()
    rmse = mse**0.5
    return rmse

rng = np.random.default_rng(SEED)
N = 500
N_tilde = 10
P = 2
K = 3
alpha = 0
sigma = 1
X = rng.normal(size=(N, P))
X = np.hstack([X, np.prod(X, axis=1).reshape((N, 1))])
X_tilde = rng.normal(size=(N_tilde, P))
X_tilde = np.hstack([X_tilde, np.prod(X_tilde, axis=1).reshape((N_tilde, 1))])
beta = np.array([1.5, 0.2, 0.5])

y = rng.normal(alpha + X @ beta, sigma)
y_tilde = rng.normal(alpha + X_tilde @ beta, sigma)

mixture_string = """
    data {
        int<lower=0> N;
        int<lower=0> N_tilde;
        int<lower=2> K;
        int<lower=1> P;
        matrix[N, P] X;
        matrix[N_tilde, P] X_tilde;
        vector[N] y;
    }

    parameters {
        vector[K] alpha;
        vector[P + 2] beta;
        real<lower=0> sigma;
        simplex[K] w;
    }

    model {
        alpha ~ normal(0, 1);
        beta ~ normal(0, 1);
        sigma ~ normal(0, 1);

        for(i in 1:N) {
            row_vector[K] lps = [
                log(w[1]) + normal_lpdf(y[i] | alpha[1] + beta[1] * X[i,1], sigma),
                log(w[2]) + normal_lpdf(y[i] | alpha[2] + beta[2] * X[i,2], sigma),
                log(w[3]) + normal_lpdf(y[i] | alpha[3] + X[i] * beta[3:], sigma)
            ];
            target += log_sum_exp(lps);
        }
    }

    generated quantities {
        vector[N_tilde] post_pred;

        for(j in 1:N_tilde) {
            row_vector[K] preds = [
                normal_rng(alpha[1] + beta[1] * X_tilde[j,1], sigma),
                normal_rng(alpha[2] + beta[2] * X_tilde[j,2], sigma),
                normal_rng(alpha[3] + X_tilde[j] * beta[3:], sigma)
            ];
            int mix_idx = categorical_rng(w);
            post_pred[j] = preds[mix_idx];
        }
    }
"""

with open("docs/user-guide/scripts/mixture.stan", "w") as stan_file:
    stan_file.write(mixture_string)

mixture = csp.CmdStanModel(stan_file="docs/user-guide/scripts/mixture.stan")

fit_mixture = mixture.sample(
    data={"N": N, "N_tilde": N_tilde, "P": P, "K": K, "X": X[:,:P], "X_tilde": X_tilde[:,:P], "y": y},
    seed=SEED
)

mixture_weights = fit_mixture.w
beta_hat = fit_mixture.beta

def density(x):
    limits = x.min(), x.max()
    grid = np.linspace(*limits, 1000)
    return grid, gaussian_kde(x)(grid)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
ax[0].errorbar(
    [rf"$\hat{{\beta}}_{i + 1}$" for i in range(beta_hat.shape[1])],
    beta_hat.mean(axis=0),
    yerr = [
        abs(beta_hat.mean(axis=0) - np.quantile(beta_hat, 0.025, axis=0)), 
        abs(-beta_hat.mean(axis=0) + np.quantile(beta_hat, 0.975, axis=0))
    ],
    fmt="o",
    markerfacecolor="none",
    color="black",
)
ax[0].scatter(
    [rf"$\hat{{\beta}}_{i + 1}$" for i in range(beta_hat.shape[1])],
    np.hstack([beta[:2], beta[:2]]),
    marker="^",
    color="blue",
    s=50,
)

fills = ["indianred", "steelblue", "mediumseagreen"]
densities = list(zip(*[density(weight) for weight in mixture_weights.T]))
normalizer = np.sum(densities[1])
for k in range(K):
    ax[1].plot(densities[0][k], densities[1][k] / normalizer, color=fills[k], linewidth=0)
    ax[1].fill_between(
        densities[0][k],
        densities[1][k] / normalizer,
        [0]*len(densities[0][k]),
        color=fills[k],
        alpha=0.5,
        label=f"$\hat{{w}}_{k + 1}$",
    )
    ax[1].legend(frameon=False)
    ax[1].set_ylabel("density")
    ax[1].set_xlabel("weight")

plt.savefig("docs/user-guide/scripts/figures/mixture-weights.png", dpi=120)
plt.close()

mixture_rmse = rmse(fit_mixture.post_pred, y_tilde)

regression_string = """
    data {
        int<lower=0> N;
        int<lower=0> N_tilde;
        int<lower=1> P;
        matrix[N, P] X;
        matrix[N_tilde, P] X_tilde;
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
        vector[N_tilde] post_pred;

        for(i in 1:N) {
            log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
        }
        
        for(j in 1:N_tilde)
            post_pred[j] = normal_rng(alpha + X_tilde[j] * beta, sigma);
    }
"""

with open("docs/user-guide/scripts/regression.stan", "w") as stan_file:
    stan_file.write(regression_string)

regression = csp.CmdStanModel(stan_file="docs/user-guide/scripts/regression.stan")

regression_predictors = [
    (X[:,[*p]], X_tilde[:,[*p]])
    for p in ([0], [1], [0, 1])
]

regression_fits = [
    regression.sample(
        data={"N": N, "N_tilde": N_tilde, "P": x.shape[1], "X": x, "X_tilde": x_tilde, "y": y}, 
        seed=SEED
    )
    for (x, x_tilde) in regression_predictors
]

pbma = bb.PseudoBma(
    {f"fit{i}": bb.Draws(log_lik=fit.log_lik) for i, fit in enumerate(regression_fits)},
    bootstrap=False,
    seed=SEED
)
pbma.fit()
pbma_blend = pbma.predict(
    model_draws={
        f"fit{i}": bb.Draws(post_pred=fit.post_pred) 
        for i, fit 
        in enumerate(regression_fits)
    }
)
pbma_rmse = rmse(pbma_blend.post_pred, y_tilde)

pbma_plus = bb.PseudoBma(
    {f"fit{i}": bb.Draws(log_lik=fit.log_lik) for i, fit in enumerate(regression_fits)},
    seed=SEED
)
pbma_plus.fit()
pbma_plus_blend = pbma_plus.predict(
    model_draws={
        f"fit{i}": bb.Draws(post_pred=fit.post_pred) 
        for i, fit 
        in enumerate(regression_fits)
    }
)
pbma_plus_rmse = rmse(pbma_plus_blend.post_pred, y_tilde)

loo_i = [
    az.loo(az.from_cmdstanpy(fit)).loo_i.values
    for fit
    in regression_fits
]
stack = bb.MleStacking.from_lpd(
        lpd = {
            f"fit{i}": loo
            for i, loo 
            in enumerate(loo_i)
        },
)
stack.fit()
stack_blend = stack.predict(
    model_draws={f"fit{i}": bb.Draws(post_pred=fit.post_pred) for i, fit in enumerate(regression_fits)}
)
stack_rmse = rmse(stack_blend.post_pred, y_tilde)

fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

scatter_kwargs = dict(facecolor="none", linewidth=2)
ax[0].scatter([1, 2, 3], mixture_weights.mean(axis=0), edgecolor="indianred", **scatter_kwargs, label="mixture model")
ax[0].scatter([1, 2, 3], list(pbma.weights.values()), edgecolor="steelblue", **scatter_kwargs, label="pseudo-BMA")
ax[0].scatter([1, 2, 3], list(pbma_plus.weights.values()), edgecolor="skyblue", **scatter_kwargs, label="pseudo-BMA+")
ax[0].scatter([1, 2, 3], list(stack.weights.values()), edgecolor="mediumseagreen", **scatter_kwargs, label="stacking")
ax[0].set_xticks([1, 2, 3])
ax[0].set_xticklabels(["model 1", "model 2", "model 3"])
ax[0].set_ylabel("weights")
ax[0].legend(frameon=True)

ax[1].scatter(
    ["mixture", "pseudo-BMA", "pseudo-BMA+", "stacking"],
    [
        mixture_rmse,
        pbma_rmse,
        pbma_plus_rmse,
        stack_rmse,
    ],
    marker="o",
    color="black",
)
ax[1].set_ylabel("RMSE (out-of-sample)")

plt.savefig("docs/user-guide/scripts/figures/stacking-compare.png", dpi=120, bbox_inches="tight")
plt.close()
