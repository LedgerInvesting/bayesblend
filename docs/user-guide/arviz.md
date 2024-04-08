# Arviz integration

BayesBlend can be used in conjunction with typical
[Arviz](
https://github.com/arviz-devs/arviz
)
workflows, while remaining flexible enough
to support bespoke modelling pipelines not dependent
on Arviz.

## The `BayesBlendModel.from_arviz` class method

We can instantiate a new `BayesBlendModel` class
from Arviz `InferenceData` objects via the
`from_arviz` class method, i.e.

```python title="arviz.InferenceData to BayesBlendModel"
import arviz as az
import bayesblend as bb
import numpy as np

# Create dummy arviz.InferenceData objects
shape = (4, 1000, 10)

idatas = {
    f"fit{i}": az.from_dict(
        log_likelihood={"log_lik": np.random.normal(size=shape)},
        posterior_predictive={"post_pred": np.random.normal(size=shape)},
    )
    for i
    in range(2)
}

# Use the `from_arviz` class method
stack = bb.MleStacking.from_arviz(idatas)
```

## Transforming back to `arviz.InferenceData` objects

You can also transform BayesBlend `Draws` to `arviz.InferenceData`
objects using the `io.Draws.to_arviz` method.
This method accepts a `dims` argument so that the `log_lik`
and `post_pred` arrays can be correctly shaped according
to Arviz's `(chains, draws, *variables)` shape requirement.
Otherwise, Arviz will warn that there are more samples
than draws.

Continuing from the previous example:

```python title="Draws.to_arviz"
stack.fit()
blend = stack.predict()

blended_idata = blend.to_arviz(dims=shape)
```

which returns an `InferenceData` object for the
blended samples:

```
Inference data with groups:
        > posterior_predictive
        > log_likelihood
```

## Using BayesBlend with estimates of log predictive densities

A common use-case of BayesBlend is blending
predictions using approximate estimates of
out-of-sample accuracy, such as [Pareto-smoothed
importance sampling leave-one-out cross-validation](
https://arxiv.org/abs/1507.04544
)
estimates of out-of-sample log predictive densities (LPD).
For instance, we could use `arviz.loo` to estimate
these values, which returns a pointwise array of
log predictive densities.
In this scenario, we don't want to use the raw
log likelihood samples directly, but our newly
estimated LPD values.

BayesBlend can also be used with these intermediate
data structures by using the `from_lpd` class method,
which takes a dictionary of LPD arrays and a dictionary
of posterior predictive arrays:

```python title="BayesBlend.from_lpd"
# Use PSIS-LOO on our InferenceData objects
loos = {
    name: az.loo(idata, reff=1)
    for name, idata
    in idatas.items()
}

# Initialize a BayesBlendModel object
stack_from_lpd = bb.MleStacking.from_lpd(
    lpd={
        name: loo.loo_i.values 
        for name, loo 
        in loos.items()
    },
    post_pred={
        name: idata.posterior_predictive["post_pred"] 
        for name, idata 
        in idatas.items()
    },
)
```
