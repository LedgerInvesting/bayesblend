data {
  int<lower=1> N; // N datapoints
  int<lower=2> M; // N models
  int<lower=0> K; // N discrete covariates
  int<lower=0> P; // N continuous covariates
  real alpha_loc;
  real<lower=0> alpha_scale;
  real<lower=0> tau_mu_global;
  real<lower=0> tau_mu_disc;
  real<lower=0> tau_mu_cont;
  real<lower=0> tau_sigma_disc;
  real<lower=0> tau_sigma_cont;
  real<lower=0> lambda_loc;
  matrix[N, M] y;
  matrix[N, K + P] X;
  int<lower=0, upper=1> adaptive;
}

transformed data{
  int<lower=1> Mm1 = M - 1;
}

parameters {
  real mu_global_prime;
  array[Mm1] real alpha_prime;
  vector<lower=0>[Mm1] sigma_disc_prime;
  vector<lower=0>[Mm1] sigma_cont_prime;
  vector[Mm1] mu_disc_prime;
  vector[Mm1] mu_cont_prime;
  matrix[Mm1, K] beta_disc_prime;
  matrix[Mm1, P] beta_cont_prime;
  array[adaptive] real<lower=0> lambda;
}

transformed parameters {
  real mu_global;
  array[Mm1] real alpha;
  vector[Mm1] mu_disc;
  vector[Mm1] mu_cont;
  vector<lower=0>[Mm1] sigma_disc;
  vector<lower=0>[Mm1] sigma_cont;
  matrix[Mm1, K] beta_disc;
  matrix[Mm1, P] beta_cont;
  real<lower=0> delta;

  // Unconstrained reference weight array
  // Log-odds of model m to reference model
  array[N] vector[M] w_star;
  // Weights array
  array[N] simplex[M] w;
  // Log likelihood
  array[N] real ll;
  // Beta matrix
  matrix[Mm1, K + P] Beta;

  if(adaptive)
    delta = N^lambda[1];
  else
    delta = 1.0;

  // global coefficient mean
  mu_global = tau_mu_global * delta * mu_global_prime;

  for(m in 1:Mm1){
    // non-centering is done here to allow for variance components to be 0
    alpha[m] = alpha_loc + alpha_scale * delta * alpha_prime[m];
    sigma_disc[m] = tau_sigma_disc * sigma_disc_prime[m]; 
    sigma_cont[m] = tau_sigma_cont * sigma_cont_prime[m];

    // model-level discrete coef mean
    mu_disc[m] = mu_global + tau_mu_disc * delta * mu_disc_prime[m];
    // model-level continuous coef mean
    mu_cont[m] = mu_global + tau_mu_cont * delta * mu_cont_prime[m];

    // beta weights for discrete covars
    beta_disc[m] = mu_disc[m] + sigma_disc[m] * delta * beta_disc_prime[m];
    // beta weights for continuous covars
    beta_cont[m] = mu_cont[m] + sigma_cont[m] * delta * beta_cont_prime[m];
    // regression model
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

  mu_global_prime ~ std_normal();
  mu_disc_prime ~ std_normal();
  mu_cont_prime ~ std_normal();
  
  alpha_prime ~ std_normal();

  to_vector(beta_disc_prime) ~ std_normal();
  to_vector(beta_cont_prime) ~ std_normal();
  sigma_disc_prime ~ std_normal();
  sigma_cont_prime ~ std_normal();

  target += ll;
}
