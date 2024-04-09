# Getting Started

This guide illustrates the basic functionality of
the `BayesBlendModel` and `Draws` classes.

--------------------------------------------------

BayesBlend provides an easy-to-use interface for combining predictions from multiple Bayesian models using techniques including (pseudo) Bayesian model averaging, stacking, and hierarchical stacking. The core functionality is divided into two classes: 

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

Each of these models takes a dictionary of `Draws` objects as input (one for each underlying substantive model of interest). The core functionality of each `BayesBlendModel` is then housed in the `.fit` and `.blend` methods. The `.fit` method fits the associated averaging/stacking model given the `Draws` fit data, and the latter returns a new `Draws` object that blends together the posterior predictions across substantive models given the estimated averaging/stacking parameters. 

## Example

The below example expands on the `cmdstanpy` [example](https://github.com/stan-dev/cmdstanpy?tab=readme-ov-file#example), fitting a bernoulli model twice and then blending the results together.

```stan title="bernoulli.stan"
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
```

```python title="Fit and stack models"
from cmdstanpy import CmdStanModel
from bayesblend import MleStacking

model = CmdStanModel(stan_file="bernoulli.stan")

stan_data = {
    "N": 10,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
}

# fit model to data (pretend these are different models!)
fit1 = model.sample(chains=4, data=stan_data, seed=1)
fit2 = model.sample(chains=4, data=stan_data, seed=2)

# fit the MleStacking model - fit() returns self
mle_stacking_fit = MleStacking.from_cmdstanpy(dict(fit1=fit1, fit2=fit2))
mle_stacking_fit.fit()

# generate blended predictions
mle_stacking_blend = mle_stacking_fit.predict()
```

The `mle_stacking_blend` object is a `bayesblend.Draws` object.

The particular weights estimated in the stacking model can be inspected
using `mle_stacking_fit.weights`. 

For models like hierarchical stacking with covariates, the weights during
prediction might differ from the weights estimated in the stacking model
because the weights have their own posterior predictive distribution.
For this instance, setting `return_weights=True` in `BayesBlendModel.predict` 
will return a tuple of the blended predictions and a dictionary of weights
used during prediction.
