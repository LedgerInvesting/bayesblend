data {
  int<lower=1> N;
  int<lower=2> M;
  vector<lower=0>[M] w_prior;
  matrix[N, M] y;
}

parameters {
  simplex[M] w;
}

transformed parameters {
  // Log likelihood
  vector[N] ll;
  for(n in 1:N)
    ll[n] = log_sum_exp(y[n,]' + log(w));
}

model {
  w ~ dirichlet(w_prior);
  target += ll;
}
