//
//  orthongalize_basis.cpp
//    Given a vector x and raw basis expansion (e.g. B-splines matrix)
//    Run Gram-Schmidt procedure to get the new basis elements
//  Created by Sameer Deshpande on 5/8/20.
//


#include<RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <stdio.h>

// [[Rcpp::export]]
Rcpp::List orthongalize_basis(arma::vec x,
                              arma::mat phi_raw)
{
  int n = phi_raw.n_rows;
  int D = phi_raw.n_cols;
  
  
  arma::mat Phi_init = arma::zeros<arma::mat>(n,D+2);
  Phi_init.col(0).fill(1.0); // column of 1's
  Phi_init.col(1) = x; // column now is of x
  Phi_init.cols(2,D+1) = phi_raw;
  
  arma::mat Phi_orth = arma::zeros<arma::mat>(n,D+2);
  // Start gram-schmidt procedure now
  Phi_orth.col(0) = Phi_init.col(0);
  arma::vec proj = arma::zeros<arma::vec>(n);
  for(int j = 1; j < D+2; j++){
    proj = Phi_init.col(j);
    for(int jj = 0; jj < j; jj++){
      proj -= arma::dot(Phi_init.col(j), Phi_orth.col(jj))/arma::dot(Phi_orth.col(jj), Phi_orth.col(jj)) * Phi_orth.col(jj);
    }
    Phi_orth.col(j) = proj;
  }
  
  Rcpp::List results;
  results["Phi_init"] = Phi_init;
  results["Phi_orth"] = Phi_orth;
  return(results);
  
  
}
