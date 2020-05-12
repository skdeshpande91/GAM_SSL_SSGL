//
//  test_update_B.cpp
//  
//
//  Created by Sameer Deshpande on 5/8/20.
//

#include<RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "../../SpikeSlabGAM/src/update_beta.h"
#include <stdio.h>

// [[Rcpp::export]]
Rcpp::List test_update_B(arma::mat B,
                         const arma::vec R,
                         const arma::cube Phi,
                         const double sigma2,
                         const double xi1,
                         const double xi0,
                         const arma::vec theta,
                         const int max_iter = 1000,
                         const double eps = 1e-9,
                         const bool verbose = false)
{
  int D = B.n_rows;
  int p = B.n_cols;
  
  int n = R.n_elem;
  arma::mat B_init = B;
  
  update_beta(B, R, Phi, sigma2, xi1, xi0, theta, max_iter, eps, n, p, D, verbose);
  
  Rcpp::List results;
  results["B_init"] = B_init;
  results["B"] = B;
  return(results);
  
  
  
}
