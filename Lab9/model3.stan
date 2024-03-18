
data {
  int<lower=0> N;
  int y[N]; //death
  vector[N] log_y; //log of y
  vector[N] x; //proportion of outside worker
}


parameters {
  vector[N] alpha;
  real beta;
  real mu;
  real <lower = 0> sigma;
}

transformed parameters{
  vector[N] log_theta;
  
  log_theta = alpha + beta*x;
}

model {
  
  alpha ~ normal(mu,sigma);
  beta ~ normal(0,1);
  
  mu ~ normal(0,1);
  sigma ~ normal(0,1);

  vector[N] log_lambda;
  for (n in 1:N) {
    log_lambda[n] = log_theta[n] + log_y[n]; // Corrected element-wise operation
  }
  // Poisson log-likelihood
  y ~ poisson_log(log_lambda);
}

generated quantities {
  vector[N] log_lambda;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lambda[n] = log_theta[n] + log_y[n];
    log_lik[n] = poisson_log_lpmf(y[n] | log_lambda[n]); // Use computed log_lambda
  }
}
