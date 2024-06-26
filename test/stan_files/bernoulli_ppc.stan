data {
    int<lower=0> N;
    array[N] int<lower=0, upper=1> y;
    real<lower=0> beta_alpha;
    real<lower=0> beta_beta;
    int<lower=0, upper=1> prior_only;
} 
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(beta_alpha, beta_beta);
    if(!prior_only)
      y ~ bernoulli(theta);
}
generated quantities {
  array[N] int post_pred;
  real log_joint_prior;
  vector[N] log_lik;

  for (n in 1:N){
    log_lik[n] = bernoulli_lpmf(y[n] | theta);
    post_pred[n] = bernoulli_rng(theta);
  }
  
  log_joint_prior = beta_lpdf(theta | 1, 1);
}
