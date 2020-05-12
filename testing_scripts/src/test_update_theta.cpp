//
//  test_theta_update.cpp
//  
//
//  Created by Sameer Deshpande on 5/8/20.
//

#include<RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "../../SpikeSlabGAM/src/update_theta.h"
#include <stdio.h>

// [[Rcpp::export]]
Rcpp::List test_theta_update(arma::vec theta_init,
                             arma::vec alpha,
                             arma::mat B,
                             const arma::vec delta,
                             const double lambda1,
                             const double lambda0,
                             const double xi1,
                             const double xi0,
                             const int max_iter = 1000,
                             const double eps = 1e-9)
{
  int D = B.n_rows;
  int p = B.n_cols;
  
  
  // eventually make this pointers/arrays since dimension is fixed anyway
  std::vector<double> alpha_spike_dens(p);
  std::vector<double> alpha_slab_dens(p);
  std::vector<double> beta_spike_dens(p);
  std::vector<double> beta_slab_dens(p);
  
  for(int j = 0; j < p; j++){
    alpha_spike_dens[j] = lambda0 * exp(-1.0 * lambda0 * abs(alpha(j)));
    alpha_slab_dens[j] = lambda1 * exp(-1.0 * lambda1 * abs(alpha(j)));
    
    beta_spike_dens[j] = xi0 * exp(-1.0 * xi0 * arma::norm(B.col(j)));
    beta_slab_dens[j] = xi1 * exp(-1.0 * xi1 * arma::norm(B.col(j)));
  }
  arma::vec theta_newton = theta_init;
  //arma::vec theta_grad = theta_init;
  update_theta(theta_newton, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p, max_iter, eps);
  
  //update_theta_grad(theta_grad, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p, max_iter, eps);

  Rcpp::List results;
  results["theta_newton"] = theta_newton;
  //results["theta_grad"] = theta_grad;
  results["alpha_spike_dens"] = alpha_spike_dens;
  results["alpha_slab_dens"] = alpha_slab_dens;
  results["beta_spike_dens"] = beta_spike_dens;
  results["beta_slab_dens"] = beta_slab_dens;
  return(results);
}

// [[Rcpp::export]]
double Q_theta(arma::vec tiltheta,
               arma::vec alpha,
               arma::mat B,
               const arma::vec delta,
               const double lambda1,
               const double lambda0,
               const double xi1,
               const double xi0)
{
  
  int p = B.n_cols;
  //arma::vec tiltheta = arma::zeros<arma::mat>(3);
  //tiltheta(0) = theta_01;
  //tiltheta(1) = theta_10;
  //tiltheta(2) = theta_11;

  std::vector<double> alpha_spike_dens(p);
  std::vector<double> alpha_slab_dens(p);
  std::vector<double> beta_spike_dens(p);
  std::vector<double> beta_slab_dens(p);
  
  for(int j = 0; j < p; j++){
    alpha_spike_dens[j] = lambda0 * exp(-1.0 * lambda0 * abs(alpha(j)));
    alpha_slab_dens[j] = lambda1 * exp(-1.0 * lambda1 * abs(alpha(j)));
    
    beta_spike_dens[j] = xi0 * exp(-1.0 * xi0 * arma::norm(B.col(j)));
    beta_slab_dens[j] = xi1 * exp(-1.0 * xi1 * arma::norm(B.col(j)));
  }
  
  double objective = theta_objective(tiltheta, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
  return(objective);
}


