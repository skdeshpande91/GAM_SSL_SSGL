library(Rcpp)
library(RcppArmadillo)

n <- 20
x <- runif(n, -1,1)
nknots <- 3
knots <- quantile(x, probs = seq(0,1, length = nknots + 1))
# bs wants only the *internal* breakpoints
phi_raw <- bs(x, knots = knots , deg = 3)
phi_raw <- phi_raw[,-ncol(phi_raw)]



sourceCpp("../SpikeSlabGAM/src/orthogonalize_basis.cpp")

test <- orthongalize_basis(x, phi_raw)





generate_phi <- function(x, nknots = 5){
  n <- length(x)
  knots <- quantile(x, probs = seq(0, 1, length = nknots+1))
  phi_raw <- bs(x, knots = knots, deg = 3)
  
  phi_tilde <- cbind(rep(1, times = n), x, phi_raw)
  
  
  phi_tilde_qr <- qr(phi_tilde)
  
  phi_tilde_Q <- qr.Q(phi_qr)
  
  phi_out <- 
  # Need to re-scale as well
}