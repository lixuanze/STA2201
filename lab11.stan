data {
  int<lower=0> N; // number of years
  int<lower=0> S; // number of states
  matrix[N, S] y; // log ent per capita
  int<lower=0> K; // number of splines
  matrix[N, K] B; // splines
}

parameters {
  matrix[K, S] alpha;
  vector<lower=0>[S] sigma_alpha;
  vector<lower=0>[S] sigma_y;
}

transformed parameters {
  matrix[N, S] mu = B * alpha; 
}

model {
  sigma_alpha ~ normal(0, 1);
  sigma_y ~ normal(0, 1);
  
  for (s in 1:S) {
    y[, s] ~ normal(mu[, s], sigma_y[s]);
    
    // Priors for alpha
    alpha[1, s] ~ normal(0, sigma_alpha[s]);
    alpha[2, s] ~ normal(alpha[1, s], sigma_alpha[s]);
    for (k in 3:K) {
      alpha[k, s] ~ normal(2 * alpha[k-1, s] - alpha[k-2, s], sigma_alpha[s]);
    }
  }
}
