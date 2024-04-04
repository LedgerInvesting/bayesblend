# Simulated blending weights via mixture modelling

*This guide introduces a simple example of recovering blending
weights from a set of simulated models.*

----------------------------------------------------------

The simplest example to demonstrate the utility of BayesBlend,
and model blending more generally, is to use simulated univariate data
of size $N$ from a set of $K$ models, $\mathcal{M} = \{M_{1}, M_{2}, ..., M_{k}\}$,
with a known $N \times K$ matrix of mixture weights, $\mathbf{W}$.
Indeed, model averaging via stacking is akin to fitting a mixture model
across $\mathcal{M}$, rather than the two-step stacking process
(see section 4.3 of
[Yao *et al.* (2018)](http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf)
for more information).
This replicates the $\mathcal{M}$-closed setting, as described by
[Bernado & Smith (1994)](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870),
[Yao *et al.* (2018)](http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf)
and others,
where the true model is known, in this case being a 
mixture of the different candidate models in $\mathcal{M}$.

## Example models

For this simple example, we'll use the following 3 linear regression models, which
vary only by the specification of the linear predictor.
Specifically, we have two predictors that, in a real application,
we might be unsure of their exact relationship with the response variable
$\mathbf{y}$.
Each model uses a Gaussian likelihood distribution and weakly informative
priors.

Model 1 includes a single predictor $\mathbf{x}_{1}$:

\begin{align}
    \tag{Model 1}
    y_{i} &\sim \mathrm{Normal}(\mu_{i}, \sigma)\\
    \mu_{i} &= \alpha + \beta x_{i1}\\
    \alpha &\sim \mathrm{Normal}(0, 1)\\
    \beta &\sim \mathrm{Normal}(0, 1)\\
    \sigma &\sim \mathrm{Normal}^{+}(0, 1)\\
\end{align}

Model 2 includes a single predictor $\mathbf{x}_{2}$:

\begin{align}
    \tag{Model 2}
    y_{i} &\sim \mathrm{Normal}(\mu_{i}, \sigma)\\
    \mu_{i} &= \alpha + \beta x_{i2}\\
    \alpha &\sim \mathrm{Normal}(0, 1)\\
    \beta &\sim \mathrm{Normal}(0, 1)\\
    \sigma &\sim \mathrm{Normal}^{+}(0, 1)\\
\end{align}

Model 3 includes both predictors:

\begin{align}
    \tag{Model 3}
    y_{i} &\sim \mathrm{Normal}(\mu_{i}, \sigma)\\
    \mu_{i} &= \alpha + \beta_{1} x_{i1} + \beta_{2} x_{i2}\\
    \alpha &\sim \mathrm{Normal}(0, 1)\\
    \beta_{1} &\sim \mathrm{Normal}(0, 1)\\
    \beta_{2} &\sim \mathrm{Normal}(0, 1)\\
    \sigma &\sim \mathrm{Normal}^{+}(0, 1)\\
\end{align}

## Simulating the data 

```python title="Data simulation"
import numpy as np
import cmdstanpy as csp

import bayesblend as bb

SEED = 1234

rng = np.random.default_rng(SEED)

# Set the simulation constants
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

# Use numpy's random.choice to select models
# according to the weights, W
y = np.array([
    rng.normal(mus[idx - 1, i], sigma)
    for i, idx
    in enumerate(rng.choice(range(K), p=W, size=N))
])
```

## Fitting the models

```stan title="regression.stan"
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
```

```python title="Fit the models with cmdstanpy"
model = csp.CmdStanModel(stan_file="regression.stan")

fits = [
    model.sample(data={"N": N, "P": x.shape[1], "X": x, "y": y})
    for x in (X[:,0].reshape((100, 1)), X[:,1].reshape((100, 1)), X)
]
```

## Recovering the mixture weights
