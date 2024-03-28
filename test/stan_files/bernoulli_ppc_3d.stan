data {
    int<lower=0> N;
    int<lower=1> D;
    array[N,D] int<lower=0, upper=1> y;
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
    for (d in 1:D){
      y[1:N,d] ~ bernoulli(theta);
    }
}
generated quantities {
  array[N,D] int post_pred;
  real log_joint_prior;
  array[N,D] real log_lik;

  for (n in 1:N){
    for (d in 1:D){
      log_lik[n,d] = bernoulli_lpmf(y[n,d] | theta);
      post_pred[n,d] = bernoulli_rng(theta);
    }
  }
  
  log_joint_prior = beta_lpdf(theta | 1, 1);
}
