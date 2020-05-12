# First test of update_B

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
X_test_orig <- matrix(runif(n_test*p, 0,1), nrow = n_test, ncol = p)

D <- 10

tmp_data <- prepare_X_Phi(X_train_orig, X_test_orig, D = D, spline_type = "n")
n <- n_train
X <- tmp_data$X_train
Phi <- tmp_data$Phi_train

#######
# Generate some data
########
sigma <- 0.75
B_true <- matrix(0,nrow = D-1, ncol = p)
B_true[,1] <- runif(D-1, -10,10)
B_true[,3] <- runif(D-1, -10,10)
B_true[,5] <- runif(D-1, -10,10)
R <- Phi[,,1] %*% B_true[,1] + Phi[,,3] %*% B_true[,3] + Phi[,,5] %*% B_true[,5] + sigma * rnorm(n,0,1)


# Try a few different values of xi0
xi1 <- 1
sourceCpp("src/test_update_B.cpp")
B_init <- matrix(0, nrow = D-1, ncol = p)
theta <- c(0.25,0.25,0.25, 0.25)

norm <- function(x){sqrt(sum(x*x))}
B_true_norm <- apply(B_true, MARGIN = 2, FUN = norm)
test_0 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 1 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm0 <- apply(test_0$B, MARGIN = 2, FUN = norm)
range(B_norm0[B_norm0 != 0 & B_true_norm != 0])
range(B_norm0[B_norm0 != 0 & B_true_norm == 0])

test_1 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 10 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm1 <- apply(test_1$B, MARGIN = 2, FUN = norm)
sum(B_norm1 != 0 & B_true_norm == 0)
range(B_norm1[B_norm1 != 0 & B_true_norm != 0])
range(B_norm1[B_norm1 != 0 & B_true_norm == 0])


test_3 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 100 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm3 <- apply(test_3$B, MARGIN = 2, FUN = norm)
sum(B_norm3 != 0 & B_true_norm == 0)
range(B_norm3[B_norm3 != 0 & B_true_norm != 0])


test_4 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 500 * sqrt(D-1) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm4 <- apply(test_4$B, MARGIN = 2, FUN = norm)
sum(B_norm4 != 0 & B_true_norm == 0)
sum(B_norm4 == 0 & B_true_norm != 0)
