from cmdstanpy import CmdStanModel
from bayesblend import MleStacking

stan_string = """
    data {
        int<lower=0> N;
        array[N] int<lower=0, upper=1> y;
    } 
    parameters {
        real<lower=0, upper=1> theta;
    }
    model {
        theta ~ beta(1, 1);
        y ~ bernoulli(theta);
    }
    generated quantities {
        array[N] int post_pred;
        vector[N] log_lik;

        for (n in 1:N){
            log_lik[n] = bernoulli_lpmf(y[n] | theta);
            post_pred[n] = bernoulli_rng(theta);
        }
    }
"""

with open("bernoulli.stan", "w") as f:
    f.write(stan_string)

model = CmdStanModel(stan_file="bernoulli.stan")

stan_data = {
    "N": 10,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
}

# fit model to data (pretend these are different models!)
fit1 = model.sample(chains=4, data=stan_data, seed=1)
fit2 = model.sample(chains=4, data=stan_data, seed=2)

# fit the MleStacking model
mle_stacking_fit = MleStacking.from_cmdstanpy(dict(fit1=fit1, fit2=fit2)).fit()

# generate blended predictions
mle_stacking_blend = mle_stacking_fit.predict()
