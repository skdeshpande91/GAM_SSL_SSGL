library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/test_theta_update.cpp")

set.seed(129)
p <- 500
D <- 10

alpha <- rep(0, times = p)
alpha[1:3] <- rnorm(3, 0, 10)

B <- matrix(0, nrow = D, ncol = p)
B[,1:5] <- rnorm(5*D, 0, 10)

####################################
# Create some initial hyperparameters
lambda1 <- 0.01
lambda0 <- 100
xi1 <- 0.01
xi0 <- 100
delta <- rep(1, times = 4)
###################################

##################################
# Compute true spike and slab densities

alpha_spike_dens <- lambda0 * exp(-1 * lambda0 * abs(alpha))
alpha_slab_dens <- lambda1 * exp(-1 * lambda1 * abs(alpha))

beta_spike_dens <- rep(NA, times = p)
beta_slab_dens <- rep(NA, times = p)
for(j in 1:p){
  beta_spike_dens[j] <- xi0 * exp(-1 * xi0 * sqrt(sum(B[,j]^2)))
  beta_slab_dens[j] <- xi1 * exp(-1 * xi1 * sqrt(sum(B[,j]^2)))
}
##################################

theta <- rep(0.25, times = 4)

sourceCpp("src/test_theta_update.cpp")

test_optim <- optim(par = theta[2:4], fn = Q_theta,
                     alpha = alpha, B = B, delta = delta, lambda1 = lambda1, 
                     lambda0 = lambda0, xi1 = xi1, xi0 = xi0,
                     method = "L-BFGS-B", lower = 1e-16, upper = 1 - 1e-16,
                     hessian = TRUE)



optim_theta <- c(1 - sum(test_optim$par),test_optim$par)




test <- test_theta_update(theta, alpha, B, delta, lambda1, lambda0, xi1, xi0, eps = 1e-12)
test2 <- test_theta_update(optim_theta, alpha, B, delta, lambda1, lambda0, xi1, xi0, eps = 1e-12)

Q_theta(optim_theta[2:4], alpha, B, delta, lambda1, lambda0, xi1, xi0)
Q_theta(test$theta_newton[2:4], alpha, B, delta, lambda1, lambda0, xi1, xi0)


Q_theta(test$theta_grad[2:4], alpha, B, delta, lambda1, lambda0, xi1, xi0)


# Check that we're computing spike and slab densities alright
range(test$alpha_spike_dens - alpha_spike_dens)
range(test$alpha_slab_dens - alpha_slab_dens)
range(test$beta_spike_dens - beta_spike_dens)
range(test$beta_slab_dens - beta_slab_dens)


Q_theta(test$theta[2:4], alpha, B, delta, lambda1, lambda0, xi1, xi0)


test2 <- test_theta_update(optim_theta, alpha, B, delta, lambda1, lambda0, xi1, xi0, max_iter = 1000, eps = 1e-09)
