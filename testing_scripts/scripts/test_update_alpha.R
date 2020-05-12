library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(splines)
source("scripts/prepare_X_Phi.R")

set.seed(129)
n_train <- 50
n_test <- 50
p <- 500
X_train_orig <- matrix(runif(n_train*p, -1,1), nrow = n_train, ncol= p)
X_test_orig <- matrix(runif(n_test*p, -1,1), nrow = n_test, ncol = p)

D <- 10

tmp_data <- prepare_X_Phi(X_train_orig, X_test_orig, D = 10, spline_type = "n")
n <- n_train
X <- tmp_data$X_train
Phi <- tmp_data$Phi_train


#######
# Generate some fake data to test SSL updates
#######
alpha_true <- rep(0, times= p)
alpha_true[1:5] <- runif(5, -5,5)
sigma <- 1
R <- X %*% alpha_true + sigma * rnorm(n, 0, 1)

lambda1 <- 1
lambda0 <- 100

theta <- c(0.25,0.5,0.1, 0.15)
alpha_init <- rep(0, times = p)

sourceCpp("src/test_update_alpha.cpp")


# Start from zero and use lambda1 == lambda0
test1 <- test_update_alpha(alpha_init, R, X, sigma, lambda1, lambda1, theta)
which(test1$alpha != 0)

# Look at the range of false positives
range(test1$alpha[test1$alpha != 0 & alpha_true == 0])
# It's all quite small

test2 <- test_update_alpha(runif(p,-5,5), R, X, sigma, lambda1, lambda1, theta)
which(test2$alpha != 0)
range(test2$alpha[test2$alpha != 0 & alpha_true == 0])

test3 <- test_update_alpha(alpha_init, R, X, sigma, lambda1, 10*lambda1, theta)
which(test3$alpha != 0)
range(test3$alpha[test3$alpha != 0 & alpha_true == 0])


test4 <- test_update_alpha(runif(p, -5,5), R, X, sigma, lambda1, 10*lambda1, theta)
which(test4$alpha != 0)


test5 <- test_update_alpha(alpha_init, R, X, sigma, lambda1, 100*lambda1, theta)
which(test5$alpha != 0)

test6 <- test_update_alpha(runif(p, -5,5), R, X, sigma, lambda1, 100*lambda1, theta)
which(test6$alpha != 0)
range(test6$alpha[test6$alpha != 0 & alpha_true == 0])
