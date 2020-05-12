library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(splines)
source("scripts/prepare_X_Phi.R")

set.seed(129)
n_train <- 100
n_test <- 100
p <- 200
X_train_orig <- matrix(runif(n_train*p, 0,1), nrow = n_train, ncol= p)
X_test_orig <- matrix(runif(n_test*p, -0,1), nrow = n_test, ncol = p)

D <- 5

tmp_data <- prepare_X_Phi(X_train_orig, X_test_orig, D = D, spline_type = "n")
n <- n_train
X <- tmp_data$X_train

# orthogonalize Phi again... there's probably some weird rounding error stuff going on here
tmp_orth_phi <- orthogonalize_Phi(tmp_data$Phi_train)

Phi <- tmp_orth_phi$Phi_tilde

########
R <- 5 * sin(pi * X_train_orig[,1]) + 2.5 * (X_train_orig[,3]^2 - 0.5) + 
  1*exp(X_train_orig[,4]) + 3 * X_train_orig[,5] + sigma * rnorm(n, 0, 1)



##########
# Similar to the example from SSGL package
##########
sigma <- 1
#R <- 10 * sin(pi * X_train_orig[,1]) + 25 * (X_train_orig[,3]^2 - 0.5) + 
# 10 * exp(X_train_orig[,4]) + sigma * rnorm(n, 0, 1)

#B_true <- matrix(0,nrow = D-1, ncol = p)
#B_true[,1] <- runif(D-1, -5,5)
#B_true[,3] <- runif(D-1, -5,5)
#B_true[,5] <- runif(D-1, -5,5)

#R <- sin(pi*X_train_orig[,1]) + 2.5 * (X_train_orig[,3]^2 - 0.5) + exp(X_train_orig[,4]) + 3 * X_train_orig[,5] + sigma * rnorm(n,0,1)
#R <- Phi[,,1] %*% B_true[,1] + Phi[,,3] %*% B_true[,3] + Phi[,,5] %*% B_true[,5] + sigma * rnorm(n,0,1)
#R <- 5 * X_train_orig[,1]^2 -8 * log(X_train_orig[,2]^2 + 1) + 10 * sin(pi * X_train_orig[,3]) + sigma * rnorm(n,0,1)



xi1 <- 1

norm <- function(x){sqrt(sum(x*x))}


sourceCpp("src/test_update_B.cpp")
B_init <- matrix(0, nrow = D-1, ncol = p)
theta <- c(0.25,0.5,0.1, 0.15)
#########

test_0 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 10 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm0 <- apply(test_0$B, MARGIN = 2, FUN = norm)

#######

test_0 <- test_update_B(B_init, R, Phi, sigma*sigma, xi1,  * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_0$B, MARGIN = 2, FUN = norm)[1:10]

test_1 <- test_update_B(B_init, R, Phi, sigma*sigma, xi1, 5 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_1$B, MARGIN = 2, FUN = norm)[1:10]

test_2 <- test_update_B(test_1$B, R, Phi, sigma*sigma, xi1, 10 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_2$B, MARGIN = 2, FUN = norm)[1:10]

test_3 <- test_update_B(test_2$B, R, Phi, sigma*sigma, xi1, 25 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_3$B, MARGIN = 2, FUN = norm)[1:10]

test_4 <- test_update_B(test_3$B, R, Phi, sigma*sigma, xi1, 50 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_4$B, MARGIN = 2, FUN = norm)[1:10]

test_5 <- test_update_B(test_4$B, R, Phi, sigma*sigma, xi1, 75 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_5$B, MARGIN = 2, FUN = norm)[1:10]

test_6 <- test_update_B(test_5$B, R, Phi, sigma*sigma, xi1, 75 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
apply(test_6$B, MARGIN = 2, FUN = norm)[1:10]
# We only do ok...
