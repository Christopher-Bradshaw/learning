// Given some data y (which in truth has 0 mean and 0.1 sd) we can optimize and
// Sample from the distribution of mu/sigma
data {
  int<lower=0> N;
  real y[N];
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  y ~ normal(mu, sigma);
  // print("mu: ", mu, " sigma: ", sigma, " log likelihood: ", target());
}
