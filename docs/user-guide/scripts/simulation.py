# mypy: ignore-errors

import argparse
from typing import Optional, Tuple
import arviz as az
import numpy as np
import cmdstanpy as csp
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import warnings
from tqdm import tqdm

import bayesblend as bb

SEED = 1234
K = 3
P = 2
N = 50
N_tilde = 50
rng = np.random.default_rng(SEED)

Data = Tuple[np.ndarray, np.ndarray]


def simulate_data(seed: Optional[int] = None) -> Tuple[Data, Data]:
    alpha = rng.normal()
    sigma = 1
    X = rng.normal(size=(N, P))
    X_tilde = rng.normal(size=(N_tilde, P))
    y = rng.normal(alpha, sigma, size=N)
    y_tilde = rng.normal(alpha, sigma, size=N_tilde)

    return ((y, X), (y_tilde, X_tilde))


mixture_string = """
    data {
        int<lower=0> N;
        int<lower=0> N_tilde;
        int<lower=2> K;
        int<lower=1> P;
        matrix[N, P] X;
        matrix[N_tilde, P] X_tilde;
        vector[N] y;
        vector[N_tilde] y_tilde;
    }

    parameters {
        vector[K] alpha;
        vector[P + 2] beta;
        simplex[K] w;
    }

    transformed parameters {
        matrix[N, K] lps;

        for(i in 1:N) {
            lps[i] = [
                log(w[1]) + normal_lpdf(y[i] | alpha[1] + beta[1] * X[i,1], 1),
                log(w[2]) + normal_lpdf(y[i] | alpha[2] + beta[2] * X[i,2], 1),
                log(w[3]) + normal_lpdf(y[i] | alpha[3] + X[i] * beta[3:], 1)
            ];
        }
    }

    model {
        alpha ~ std_normal();
        beta ~ std_normal();

        for(i in 1:N)
            target += log_sum_exp(lps[i]);
    }

    generated quantities {
        matrix[N_tilde, K] log_lik;

        for(j in 1:N_tilde) {
            log_lik[j] = [
                normal_lpdf(y_tilde[j] | alpha[1] + beta[1] * X_tilde[j,1], 1),
                normal_lpdf(y_tilde[j] | alpha[2] + beta[2] * X_tilde[j,2], 1),
                normal_lpdf(y_tilde[j] | alpha[3] + X_tilde[j] * beta[3:], 1)
            ];
        }
    }
"""

with open("docs/user-guide/scripts/mixture.stan", "w") as stan_file:
    stan_file.write(mixture_string)


def fit_mixture(train, test):
    mixture = csp.CmdStanModel(stan_file="docs/user-guide/scripts/mixture.stan")
    y, X = train
    y_tilde, X_tilde = test
    fit = mixture.sample(
        data={
            "N": len(y),
            "N_tilde": len(y_tilde),
            "P": P,
            "K": K,
            "X": X[:, :P],
            "X_tilde": X_tilde[:, :P],
            "y": y,
            "y_tilde": y_tilde,
        },
        inits=0,
        seed=SEED,
    )
    return fit


regression_string = """
    data {
        int<lower=0> N;
        int<lower=0> N_tilde;
        int<lower=1> P;
        matrix[N, P] X;
        matrix[N_tilde, P] X_tilde;
        vector[N] y;
        vector[N_tilde] y_tilde;
    }

    parameters {
        real alpha;
        vector[P] beta;
    }

    transformed parameters {
        vector[N] mu = alpha + X * beta;
    }

    model {
        alpha ~ std_normal();
        beta ~ std_normal();
        y ~ normal(mu, 1);
    }

    generated quantities {
        vector[N] log_lik;
        vector[N_tilde] log_lik_tilde;
        
        for(i in 1:N) 
            log_lik[i] = normal_lpdf(y[i] | mu[i], 1);
        
        for(j in 1:N_tilde) {
            real mu_tilde = alpha + X_tilde[j] * beta;
            log_lik_tilde[j] = normal_lpdf(y_tilde[j] | mu_tilde, 1);
        }
    }
"""

with open("docs/user-guide/scripts/regression.stan", "w") as stan_file:
    stan_file.write(regression_string)

regression = csp.CmdStanModel(stan_file="docs/user-guide/scripts/regression.stan")


def fit_regressions(train, test):
    y, X = train
    y_tilde, X_tilde = test
    predictors = [(X[:, [*p]], X_tilde[:, [*p]]) for p in ([0], [1], [0, 1])]
    fits = [
        regression.sample(
            data={
                "N": N,
                "N_tilde": N_tilde,
                "P": x.shape[1],
                "X": x,
                "X_tilde": x_tilde,
                "y": y,
                "y_tilde": y_tilde,
            },
            seed=SEED,
        )
        for (x, x_tilde) in predictors
    ]
    return fits


def blend(mixture, regressions):
    loo_i = [az.loo(az.from_cmdstanpy(fit)).loo_i.values for fit in regressions]

    loo_fits = {f"fit{i}": loo for i, loo in enumerate(loo_i)}

    pred_draws = {
        f"fit{i}": bb.Draws(
            log_lik=fit.log_lik_tilde,
        )
        for i, fit in enumerate(regressions)
    }

    mix = bb.SimpleBlend(
        {f"fit{i}": bb.Draws(log_lik=mixture.log_lik[..., i]) for i in range(K)},
        weights={f"fit{i}": w for i, w in enumerate(mixture.w.T)},
    )
    mix_blend = mix.predict()

    pbma = bb.PseudoBma.from_lpd(
        loo_fits,
        bootstrap=False,
    )
    pbma.fit()
    pbma_blend = pbma.predict(pred_draws)

    pbma_plus = bb.PseudoBma.from_lpd(loo_fits, seed=SEED)
    pbma_plus.fit()
    pbma_plus_blend = pbma_plus.predict(pred_draws)

    stack = bb.MleStacking.from_lpd(loo_fits)
    stack.fit()
    stack_blend = stack.predict(pred_draws)

    stack_bayes = bb.BayesStacking.from_lpd(loo_fits, seed=SEED)
    stack_bayes.fit()
    stack_bayes_blend = stack_bayes.predict(pred_draws)

    return (
        (mix.weights, mix_blend),
        (pbma.weights, pbma_blend),
        (pbma_plus.weights, pbma_plus_blend),
        (stack.weights, stack_blend),
        (stack_bayes.weights, stack_bayes_blend),
    )


def score(blends):
    names = (
        "mixture",
        "pbma",
        "pmba+",
        "stack",
        "stack_bayes",
    )
    return {name: draws.lpd.sum() for name, draws in zip(names, blends)}


def plot(elpds, weights):
    fig, ax = plt.subplots(2, 1, figsize=(6, 7), constrained_layout=True)
    blends = [
        "mixture",
        "pseudo-BMA",
        "pseudo-BMA+",
        "stacking (mle)",
        "stacking (bayes)",
    ]
    models = [f"model {k + 1}" for k in range(K)]
    weight_array = np.array(
        [np.array([list(w.values()) for w in weight]) for weight in weights]
    )

    for elpd in elpds:
        ax[0].scatter(
            blends,
            list(elpd.values()),
            marker="o",
            color="gray",
            facecolor="none",
            alpha=0.1,
        )
    elpd_array = np.array([list(v.values()) for v in elpds])
    means = elpd_array.mean(axis=0)
    lowers, uppers = np.quantile(elpd_array, [0.025, 0.975], axis=0)
    yerr = [means - lowers, uppers - means]
    ax[0].errorbar(
        blends,
        means,
        yerr=yerr,
        marker="o",
        ls=":",
        color="steelblue",
    )

    base = ax[1].transData
    transforms = [
        Affine2D().translate(-0.2, 0.0) + base,
        Affine2D().translate(-0.1, 0.0) + base,
        base,
        Affine2D().translate(0.1, 0.0) + base,
        Affine2D().translate(0.2, 0.0) + base,
    ]
    colors = [
        "#264653",
        "#2A9D8F",
        "#E9C46A",
        "#F4A261",
        "#E76F51",
    ]
    means = weight_array.mean(axis=0)
    lowers, uppers = np.quantile(weight_array, [0.025, 0.975], axis=0)
    for k, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        error = [mean - lower, upper - mean]
        ax[1].errorbar(
            models,
            mean.flatten(),
            yerr=[e.flatten() for e in error],
            fmt="o",
            lw=2,
            color=colors[k],
            label=blends[k],
            transform=transforms[k],
        )

    ax[0].set_ylabel("ELPD")
    ax[1].set_ylabel("weights")
    ax[1].legend(frameon=False)
    plt.savefig(
        "docs/user-guide/scripts/figures/stacking-compare.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def main(M):
    elpds = []
    weights = []
    for m in tqdm(range(M)):
        train, test = simulate_data()
        mixture = fit_mixture(train, test)
        regressions = fit_regressions(train, test)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            w_hat, blends = zip(*blend(mixture, regressions))
            uw = [ww for ww in w if issubclass(ww.category, UserWarning)]
            if any(uw):
                messages = (uwm.message.args[0] for uwm in uw)
                if any("Pareto" in message for message in messages):
                    print("skipping...")
                    continue
        elpds.append(score(blends))
        weights.append(w_hat)
    plot(elpds, weights)
    return elpds, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the mixture modelling and stacking comparison code"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    elpds, weights = main(args.M)
    n = len(elpds)
    if n:
        print(
            f"Skipped {(1 - n / args.M) * 100}% of simulations due to Pareto K warnings."
        )
