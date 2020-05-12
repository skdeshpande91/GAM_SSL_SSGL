//
//  test_update_alpha.cpp
//  
//
//  Created by Sameer Deshpande on 5/8/20.
//
#include<RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "../../SpikeSlabGAM/src/update_alpha.h"
#include <stdio.h>

// [[Rcpp::export]]
Rcpp::List test_update_alpha(arma::vec alpha,
                             const arma::vec R,
                             const arma::mat X,
                             const double sigma2,
                             const double lambda1,
                             const double lambda0,
                             const arma::vec theta,
                             const int max_iter = 1000,
                             const double eps = 1e-9,
                             const bool verbose = false)
{
  int n = X.n_rows;
  int p = X.n_cols;
  
  arma::vec alpha_init = alpha;
  
  update_alpha(alpha, R, X, sigma2, lambda1, lambda0, theta, max_iter, eps, n, p, verbose);
  
  
  
  Rcpp::List results;
  results["alpha_init"] = alpha_init;
  results["alpha"] = alpha;
  return(results);
  
  
}
