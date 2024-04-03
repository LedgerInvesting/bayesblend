# Simulated blending weights via mixture modelling

*This guide introduces a simple example of recovering blending
weights from a set of simulated models.*

----------------------------------------------------------

The simplest example to demonstrate the utility of BayesBlend,
and model blending more generally, is to use simulated data
of size $N$ from a set of $K$ models, $\mathcal{M} = \{M_{1}, M_{2}, ..., M_{k}\}$,
with a known $N \times K$ matrix of mixture weights, $\mathbf{W}$.
This replicates the $\mathcal{M}$-closed setting, as described by
[Bernado & Smith (1994)](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870),
[Yao *et al.* (2018)](http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf)
and others,
where the true model is known, in this case being a 
mixture of the different candidate models in $\mathcal{M}$.

## Example models

For this example, we'll use the following 3 linear regression models, which
vary only by the specification of the linear predictor.
Each model uses a Gaussian likelihood distribution and weakly informative
priors.

### Model 1

### Model 2

### Model 3

## Simulating the data 

## Fitting the models

## Recovering the mixture weights
