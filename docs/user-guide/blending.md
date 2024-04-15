# Model averaging, stacking and blending

*This guide clarifies our use of terms such as
__model averaging__, __Bayesian model averaging__,
__stacking__ and __blending__.*

-------------------------------------------------------

The process of *model averaging*, which is part of
[ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning),
means to take a weighted 
average of (posterior) predictions from a set of $K$ models, 
$\mathcal{M} = \{M_{k}, k = 1, ..., K\}$.
Currently, there are a number of methods used to derive
model weights to average predictions.

[Bayesian model averaging](
https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
)
(BMA) refers to the process of averaging
Bayesian models using posterior model probabilities:

\begin{equation}
    p(M_{k} \mid y) = \frac{
            p(y \mid M_k) p(M_k)
        }{
           \sum_{k=1}^{K} p(y \mid M_k) p(M_k)
        }
\end{equation}

The posterior model probabilities are then used to average
predictions of new data $\tilde{y}$: 

\begin{equation}
    p(\tilde{y} \mid y) = \sum_{k=1}^{K} p(\tilde{y} \mid M_{k}) p(M_{k} \mid y)
\end{equation}

where $p(y \mid M_k) = \int p(y \mid \theta_k) p(\theta_k \mid M_k) p(M_k) d\theta_k$
for unknown model parameters $\theta_k$ in model $k$ is the integrated or marginal likelihood
for model $k$. Typically, BMA proceeds by estimating the posterior
distribution of each model separately, recognizing that
$p(M_k \mid y) \propto p(y \mid M_k) p(M_k)$, choosing a suitable
value for prior model probability $p(M_k)$ (e.g. uniform values), and renormalizing
to obtain posterior model probabilities. Alternatively, one can
obtain posterior model probabilities from the Bayes factor (the ratio
of marginal likelihoods) with user-chosen prior model probabilities
([Hoeting *et al*., 1999](file:///Users/cmgoold/Downloads/1009212519.pdf)].

There are two aspects of BMA that have made other approaches preferable:
1) marginal likelihoods for each candidate model can be difficult
to calculate and 2) BMA allocates weights as if the candidate
models were the only plausible models. Paraphrasing [Kruschke (2011)](
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=36edd08030b28d7b549e7c39c630e051e231bd98),
if Bayesian parameter estimation is re-allocating credibility across
possible parameter values, BMA is allocating credibility across
possible hypotheses or models.
This is a reasonable approach if the set of candidate models
contain the true model, or something close to the true model, 
but might not be if the true model is outside of the candidate
model set. The former is an $\mathcal{M}$-closed problem,
where the latter is an $\mathcal{M}$-open problem
([Bernado & Smith 1999](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870),
[Clyde & Iverson, 2012](file:///Users/cmgoold/Downloads/AdrianSmithVol-M-Open%20(1).pdf)).
In the $\mathcal{M}$-open setting, BMA will place 
100% weight on the most credible of candidate models
as the amount of data $N \to \infty$, as would be expected
by Bayesian inference generally.

What are we to do in the $\mathcal{M}$-open setting? 
The now-standard approach is to derive weights not from the training data
but using a measure of the generalization error of a model,
such as cross-validation (e.g.
[Wolpert, 1992](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)).
[Yao *et al*., 2018](
http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf
) introduced 
*pseudo-Bayesian model averaging* (pseudo-BMA) 
as a method of deriving weights using the expected log pointwise
predictive densities (ELPD) for each model obtained via cross-validation
with [PSIS-LOO](https://arxiv.org/abs/1507.04544).
The ELPD values are normalized to obtain a set of weights that sum to 1.
In addition, *pseudoa-Bayesian model averaging plus* (pseudo-BMA+)
accounts for uncertainty in the information criteria by applying
the Bayesian bootstrap to log likelihood vectors.

[Stacking](
https://en.wikipedia.org/wiki/Ensemble_learning#Stacking
)
is an alternative method of deriving model weights in the
$\mathcal{M}$-open setting
by solving an optimization problem. Specifically, 
the weights, $\hat{w}$, are the solution to:

\begin{equation}
    \tag{Stacking}
    \hat{w} = \mathrm{arg} \min_{w} \sum_{i=1}^{N} \sum_{k=1}^{K} w_{k} f(y_{i}, p(\Theta_{k} \mid \mathbf{y}))
\end{equation}

where $y_{i}$ is the (potentially out-of-sample) observed data, 
$w_{k}$ is the weight for model $k$,
and $f(y_{i}, p(\theta_{k} \mid \mathbf{y})$
represents any cross-validation based scoring rule
used to evaluate data point $y_{i}$ from
the posterior distribution of model $k$, $p(\Theta_{k} \mid \mathbf{y})$,
with parameters $\Theta_{k}$.
While stacking has a long history
([Wolpert, 1992](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)),
[Yao *et al*., 2018](
http://www.stat.columbia.edu/~gelman/research/published/stacking_paper_discussion_rejoinder.pdf
) generalized stacking in a Bayesian context.

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
[Yao *et al.*, 2021](https://arxiv.org/abs/2101.08954))
because of their use of a fully Bayesian model
to estimate the weights.

## Blending

In BayesBlend, we use the term *blending* to refer all different types of model averaging processes,
and to emphasise the goal of blending posterior predictions from multiple models
into a coherent distribution. Alternative software can fit models to obtain stacking
weights (although BayesBlend is the only software to implement hierarchical stacking
at the time of writing), but BayesBlend makes it easy to combine predictions
using a set of model weights.
