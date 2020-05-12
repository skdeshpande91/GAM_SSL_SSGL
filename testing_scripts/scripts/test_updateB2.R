# Second test of update B
# Create the Phi matrix using the code from the SSGL package example

library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(splines)

set.seed(129)
n <- 100
p <- 200
X <- matrix(runif(n_train*p, 0,1), nrow = n_train, ncol= p)

D <- 2
Phi <- array(dim = c(n, D, p))
for(j in 1:p){
  splineTemp <- splines::ns(X[,j], df = D)
  tmp_Phi <- matrix(nrow = n, ncol= D)
  tmp_Phi[,1] <- splineTemp[,1]
  for(jj in 2:D){
    tmpY <- splineTemp[,jj]
    tmpX <- tmp_Phi[,1:(jj-1)]
    modX <- lm(tmpY ~ tmpX)
    tmp_Phi[,jj] <- modX$residuals
  }
  
  # center and scale tmp_Phi
  tmp_col_means <- apply(tmp_Phi, FUN = mean, MARGIN = 2)
  tmp_col_sd <- apply(tmp_Phi, FUN = sd, MARGIN = 2)
  for(jj in 1:D){
    tmp_Phi[,jj] <- (tmp_Phi[,jj] - tmp_col_means[jj])/(tmp_col_sd[jj] * sqrt(n-1))
  }
  
  Phi[,,j] <- tmp_Phi
  
}

####
# Generate data
####
sigma <- 0.75
R <- sin(pi*X[,1]) + 2.5 * (X[,3]^2 - 0.5) + 
  exp(X[,4]) + 3 * X[,5] + sigma * rnorm(n,0,1)


xi1 <- 1
sourceCpp("src/test_update_B.cpp")
B_init <- matrix(0, nrow = D, ncol = p)
theta <- c(0.25,0.5,0.1, 0.15)

norm <- function(x){sqrt(sum(x*x))}

test_0 <- test_update_B(B_init, R, Phi,sigma*sigma, xi1, 5 * sqrt(D) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm0 <- apply(test_0$B, MARGIN = 2, FUN = norm)
which(B_norm0 != 0)

test_1 <- test_update_B(test_0$B, R, Phi, sigma*sigma, xi1, 100 * sqrt(D) * xi1, theta, verbose = FALSE, max_iter = 5000)
B_norm1 <- apply(test_1$B, MARGIN = 2, FUN = norm)
which(B_norm1 != 0)
