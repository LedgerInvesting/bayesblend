# Model averaging, stacking and blending

*This guide clarifies our use of the terms
__model averaging__, __Bayesian model averaging__,
__stacking__ and __blending__.*

-------------------------------------------------------

The process of *model averaging*, part of
[ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning),
means to take a weighted 
average of (posterior) predictions from a set of $K$ models, 
$\mathcal{M} = \{M_{1}, M_{2}, ..., M_{K}\}$.
Currently, there are a number of methods used to derive
model weights to average predictions.

[Bayesian model averaging](
https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
)
(BMA) refers to the process of averaging
Bayesian models using the marginal posterior probabilities as weights.
The marginal posterior probabilities, also known as posterior model probabilities,
are derived from the denominator of Bayes' rule, i.e. the
marginal likelihood or evidence for each model in $\mathcal{M}$.
Because this quantity is often difficult to calculate,
*pseudo Bayesian model averaging* (pseudo-BMA) has been introduced
as a method of approximating BMA using information criteria.

[Stacking](
https://en.wikipedia.org/wiki/Ensemble_learning#Stacking
)
is an alternative method of deriving model weights
by solving an optimization problem. Specifically, 
the weights, $\hat{w}$, are the solution to:

\begin{equation}
    \tag{Stacking}
    \hat{w} = \mathrm{arg} \min_{w} \sum_{i=1}^{N} \sum_{k=1}^{K} w_{k} f(y_{i}, p(\Theta_{k} \mid \mathbf{y}))
\end{equation}

where $y_{i}$ is the (potentially out-of-sample) observed data, 
$w_{k}$ is the weight for model $k$,
and $f(y_{i}, p(\theta_{k} \mid \mathbf{y})$
represents any scoring rule
used to evaluate data point $y_{i}$ from
the posterior distribution of model $k$, $p(\Theta_{k} \mid \mathbf{y})$,
with parameters $\Theta_{k}$.
In practice, we only need to estimate $K - 1$ weights as the final
weight is known due to $\sum_{k=1} w = 1$.

The appeal of stacking, apart from its reported improved predictive
accuracy over other procedures (see e.g. 
[Clarke, 2004](https://www.jmlr.org/papers/volume4/clarke03a/clarke03a.pdf),
[Sill *et al.*, 2009](https://arxiv.org/abs/0911.0460),
or
[Yao *et al*., 2018](
http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf
)), is its extensibility. For instance, the weights can vary over data points $i = (1, ..., N)$,
creating an $N \times K$ weight matrix $\mathbf{W}$ to be optimized.
The weights can also be a function of covariates,
and/or can be estimated hierarchically. 
These extensions have generally been referred to
as hierarchical Bayesian stacking (see
[Yao *et al.*, 2021](https://arxiv.org/abs/2101.08954)).

## Blending

We use the term *blending* to refer all different types of model averaging processes.
It is neither a specific technique (e.g. stacking or BMA) or even a
particular process such as averaging, but maintains focus on the task of blending
a set of models' predictions into a coherent predictive distribution.
