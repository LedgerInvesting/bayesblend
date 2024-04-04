# BayesBlend

BayesBlend provides an easy-to-use interface for Bayesian model averaging and Bayesian stacking. The core functionality is divided into two classes: 

- `Draws`
- `BayesBlendModel`

## `Draws`

The `Draws` class is used to store and manipulate MCMC draws from the posterior distribution of an arbitrary Bayesian model. Specifically, `Draws.log_lik` and `Draws.post_pred` indicate the posterior log likelihood and posterior predictions for each datapoint in a model. Both attributes should be in the form of an `np.ndarray` with any shape (athough their shapes should match). 

## `BayesBlendModel`

The `BayesBlendModel` is an abstract base class that is subclassed into the following various Bayesian model averaging and Bayesian stacking models: 

- `MleStacking`
- `BayesStacking`
- `HierarchicalBayesStacking`
- `PseudoBma`

Each of these models takes a dictionary of `Draws` objects as input (one for each underlying substantive model of interest). The core functionality of each `BayesBlendModel` is then housed in the `.fit` and `.predict` methods. The `.fit` method fits the associated averaging/stacking model given the `Draws` fit data, and the latter returns a new `Draws` object that blends together the posterior predictions across substantive models given the estimated averaging/stacking parameters. 

## Example

The below example expands on the `cmdstanpy` example [here](https://github.com/stan-dev/cmdstanpy?tab=readme-ov-file#example), fitting a Bernoulli model twice and then blending the results together.

```python
# simple bernoulli model
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

# write out model string to .stan file
with open("bernoulli.stan", "w") as f:
    f.write(stan_string)

# instantiate a model
stan_model = CmdStanModel(stan_file="bernoulli.stan")

# data for the model
stan_data = {
    "N": 10,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
}

# fit model to data (pretend these are different models!)
fit1 = stan_model.sample(chains=4, data=stan_data, seed=1)
fit2 = stan_model.sample(chains=4, data=stan_data, seed=2)

# initialize and fit the MleStacking model
blend_model = MleStacking.from_cmdstanpy(dict(fit1=fit1, fit2=fit2))
blend_model.fit()

# generate blended predictions
blended_draws = blend_model.predict()
```
