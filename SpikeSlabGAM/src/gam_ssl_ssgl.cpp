//
//  gam_ssl_ssgl.cpp
//  
//
//  Created by Sameer Deshpande on 5/15/20.
//

#include<RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "update_alpha.h"
#include "update_beta.h"
#include "update_theta.h"
#include<ctime>
#include <stdio.h>


// [[Rcpp::export]]
Rcpp::List gam_ssl_ssgl(arma::vec Y,
                        arma::mat X,
                        arma::cube Phi,
                        double lambda1,
                        double lambda0,
                        double xi1,
                        double xi0,
                        arma::vec delta,
                        const int max_iter = 1000,
                        const double eps = 1e-9,
                        const bool verbose = true)
{
  int n = X.n_rows;
  int p = X.n_cols;
  int D = Phi.n_rows;
  
  
  // initialize main parameters
  arma::vec alpha = arma::zeros<arma::vec>(p);
  arma::mat B = arma::zeros<arma::mat>(D,p);
  arma::vec theta = arma::zeros<arma::mat>(4);
  theta.fill(0.25); // initialize all elements of theta at 0.25
  double sigma2 = 1.0;
  
  // initialize old values of the parameters
  arma::vec alpha_old = alpha;
  arma::mat B_old = B;
  
  
  // initialize partial & full residual vectors
  arma::vec R_alpha = arma::zeros<arma::vec>(n); // partial residuals used for updating alpha
  arma::vec R_beta = arma::zeros<arma::vec>(n); // partial residuals used for updating B
  arma::vec R = arma::zeros<arma::vec>(n); // full residuals, used to update sigma^2
  
  
  // initialize vectors to hold the spike and slab densities
  std::vector<double> alpha_spike_dens(p);
  std::vector<double> alpha_slab_dens(p);
  std::vector<double> beta_spike_dens(p);
  std::vector<double> beta_slab_dens(p);
  
  // intialize things for main loop
  int iter = 0;
  double diff = 1.0;
  double diff_alpha = 0.0;
  double diff_B = 0.0;
  
  while( (iter < max_iter) && (diff > eps) ){
    
    // save the old values
    alpha_old = alpha;
    B_old = B;
    
    // update alpha
    R_alpha = Y;
    for(int j = 0; j < p; j++) R_alpha -= Phi.slice(j) * B.col(j);
    update_alpha(alpha, R_alpha, X, sigma2, lambda1, lambda0, theta, max_iter, eps, n, p, false);
    // update beta
    R_beta = Y - X * alpha;
    update_beta(B, R_beta, Phi, sigma2, xi1, xi0, theta, max_iter, eps, n, p, D, false);
    // update theta
    
    for(int j = 0; j < p; j++){
      alpha_spike_dens[j] = lambda0 * exp(-1.0 * lambda0 * abs(alpha(j)));
      alpha_slab_dens[j] = lambda1 * exp(-1.0 * lambda1 * abs(alpha(j)));
      
      beta_spike_dens[j] = pow(xi0,D) * exp(-1.0 * xi0 * arma::norm(B.col(j)));
      beta_slab_dens[j] = pow(xi1,D) * exp(-1.0 * xi1 * arma::norm(B.col(j)));
    }
    update_theta(theta, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p, max_iter, eps);
  
    // update sigma
    R = Y - X * alpha;
    for(int j = 0; j < p; j++) R -= Phi.slice(j) * B.col(j);
    sigma2 = arma::dot(R,R)/(2.0 + (double) n);
    
    diff_alpha = arma::norm(alpha - alpha_old,2);
    diff_B = arma::norm(B - B_old,2);
    diff = sqrt(pow(diff_alpha,2) + pow(diff_B,2));
    
    if(verbose == true){
      Rcpp::Rcout << "Iter " << iter;
      Rcpp::Rcout << "  diff_alpha = " << diff_alpha;
      Rcpp::Rcout << "  diff_beta = " << diff_B;
      Rcpp::Rcout << "  diff = " << diff;
    }
    iter++;
  }
  
  Rcpp::List results;
  results["alpha"] = alpha;
  results["B"] = B;
  results["sigma2"] = sigma2;
  results["theta"] = theta;
  return(results);
  
}
