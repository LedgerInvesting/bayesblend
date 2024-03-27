data {
  int<lower=1> N;
  int<lower=2> M;
  int<lower=0> K;
  int<lower=0> P;
  real alpha_loc;
  real<lower=0> alpha_scale;
  real beta_disc_loc;
  real<lower=0> beta_disc_scale;
  real beta_cont_loc;
  real<lower=0> beta_cont_scale;
  real<lower=0> lambda_loc;
  matrix[N, M] y;
  matrix[N, K + P] X;
  int<lower=0, upper=1> adaptive;
}

transformed data{
  int<lower=1> Mm1 = M - 1;
}

parameters {
  array[Mm1] real alpha_prime;
  matrix[Mm1, K] beta_disc_prime;
  matrix[Mm1, P] beta_cont_prime;
  array[adaptive] real<lower=0> lambda;
}

transformed parameters {
  array[Mm1] real alpha;
  matrix[Mm1, K] beta_disc;
  matrix[Mm1, P] beta_cont;

  // Unconstrained reference weight array
  // Log-odds of model m to reference model
  array[N] vector[M] w_star;
  // Weights array
  array[N] simplex[M] w;
  // Log likelihood
  array[N] real ll;
  // Beta matrix
  matrix[Mm1, K + P] Beta;
  // Rescaler value for prior weights
  real<lower=0> delta; 

  if(adaptive)
    delta = N^lambda[1];
  else 
    delta = 1.0;

  for(m in 1:Mm1){
    // non-centering is done here to allow for variance components to be 0
    alpha[m] = alpha_loc + alpha_scale * delta * alpha_prime[m];
    beta_disc[m] = beta_disc_loc + beta_disc_scale * delta * beta_disc_prime[m];
    beta_cont[m] = beta_cont_loc + beta_cont_scale * delta * beta_cont_prime[m];
    
    Beta[m] = append_col(beta_disc[m], beta_cont[m]);
    w_star[,m] = to_array_1d(alpha[m] + X * Beta[m]');
  }
  w_star[,M] = rep_array(0, N);

  for(n in 1:N) 
    w[n] = softmax(to_vector(w_star[n]));
  
  for(n in 1:N)
    ll[n] = log_sum_exp(y[n,]' + log(w[n]));

}

model {
  if(adaptive)
    lambda ~ exponential(lambda_loc);
  alpha_prime ~ std_normal();
  to_vector(beta_disc_prime) ~ std_normal();
  to_vector(beta_cont_prime) ~ std_normal();
  target += ll;
}
