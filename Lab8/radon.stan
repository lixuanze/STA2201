
data {
  int<lower=0> N;
  vector[N] y; 
  vector[N] x; 
  int<lower=0> J; 
  int<lower = 0, upper = J> county[N]; 
  vector[J] u; 
}


parameters {
  vector[J] alpha;
  real beta;
  real gamma0;
  real gamma1;
  real<lower=0> sigma_y;
  real<lower=0> sigma_alpha;
}


model {
  vector[N] y_hat;
  
  for(i in 1:N){
    y_hat[i] = alpha[county[i]] + beta*x[i];
  }
  
  //likelihood
  y ~ normal(y_hat, sigma_y);
  //group level model
  alpha ~ normal(gamma0+gamma1*u, sigma_alpha);
  
  //priors
  beta ~ normal(0,1);
  gamma0 ~ normal(0,1);
  gamma1 ~ normal(0,1);
  sigma_y ~ normal(0,1);
  sigma_alpha ~ normal(0,1);
}

generated quantities{
  
}

