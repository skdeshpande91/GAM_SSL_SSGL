//
//  update_alpha.cpp
//  
//
//  Created by Sameer Deshpande on 5/1/20.
//

#include "update_theta.h"

// objective is the negative log conditional posterior density!

double theta_objective(const arma::vec &tiltheta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p)
{
  double objective = delta(0) * log(1.0 - arma::accu(tiltheta));
  objective += delta(1) * log(tiltheta(0));
  objective += delta(2) * log(tiltheta(1));
  objective += delta(3) * log(tiltheta(2));
  
  
  for(int j = 0; j < p; j++){
    objective += log( (1.0 - tiltheta(1) - tiltheta(2)) * alpha_spike_dens[j] + (tiltheta(1) + tiltheta(2)) * alpha_slab_dens[j] );
    objective += log( (1.0 - tiltheta(0) - tiltheta(2)) * beta_spike_dens[j] + (tiltheta(0) + tiltheta(2)) * beta_slab_dens[j] );
  }
  return(-1.0 * objective);
}

// we will not check whether lambda1 == lambda0, xi1 == xi0 in this function!
// That will be done outside
void update_theta(arma::vec &theta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p, const int &max_iter, const double &eps)
{
  
  arma::vec grad = arma::zeros<arma::vec>(3);
  arma::mat hess = arma::zeros<arma::mat>(3,3);
  arma::vec v = arma::zeros<arma::mat>(3); // v = -(hess^{-1}) * grad
  double tgrad_v = 0.0; // tgrad_v = grad.t() * v = arma::dot(grad,v);
  double step_size = 1.0;
  

  
  // initialize stuff needed to fill in gradient and hessian entries
  double pstar = 0.0;
  double qstar = 0.0;
  double alpha_contrib = 0.0; //alpha_contrib = [pstar/(theta_10 + theta_11) - (1 - pstar)/(1 - theta_10 - theta_11)]
  double beta_contrib = 0.0; //beta_contrib = [qstar/(theta_01 + theta_11) - (1 - qstar)/(1 - theta_01 - theta)11)]
  
  // inititalize stuff for the main Newton loop
  int iter = 0;
  int counter = 0;
  double diff = 1.0;
  bool armijo_condition = true; // Armijo condition: true if obj_new < obj_old + step_size * tgrad_v (condition in backtracking line search)
  bool in_unit_int = true; // checks that all entries of tiltheta are in [0,1]
  
  arma::vec tiltheta_new = arma::zeros<arma::vec>(3);
  tiltheta_new(0) = theta(1);
  tiltheta_new(1) = theta(2);
  tiltheta_new(2) = theta(3);
  double obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
  
  
  arma::vec tiltheta_old = arma::zeros<arma::vec>(3);
  double obj_old = obj_new;
  
  Rcpp::Rcout << "Ready to start!" << std::endl;
  
  while( (iter < max_iter) && (diff > eps) ){
    tiltheta_old = tiltheta_new;
    obj_old = obj_new;
    
    
    // Begin to compute gradient and hessian
    // We start with the gradient and hessian of log prior
    
    // derivative of log p(theta) wrt to theta_01
    grad(0) = -1.0 * delta(1)/tiltheta_old(0) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_01/theta_01 - delta_00/theta_00
    // derivative of log p(theta) wrt to theta_10
    grad(1) = -1.0 * delta(2)/tiltheta_old(1) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_10/theta_10 - delta_00/theta_00
    // derivative of log p(theta) wrt to theta_11
    grad(2) = -1.0 * delta(3)/tiltheta_old(2) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_11/theta_11 - delta_00/theta_00

    hess(0,0) = delta(1)/pow(tiltheta_old(0),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old), 2);
    hess(0,1) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
    hess(0,2) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
    hess(1,1) = delta(2)/pow(tiltheta_old(1),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
    hess(1,2) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
    hess(2,2) = delta(3)/pow(tiltheta_old(2),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
    
    // now we loop over the covariate groups and add in contributions from the marginal likelihoods p(alpha | theta) and p(beta | theta)
    for(int j = 0; j < p; j++){
      
      pstar = 1.0/(1.0 + (tiltheta_old(1) + tiltheta_old(2))/(1.0 - tiltheta_old(1) - tiltheta_old(2)) * alpha_spike_dens[j]/alpha_slab_dens[j]);
      qstar = 1.0/(1.0 + (tiltheta_old(0) + tiltheta_old(2))/(1.0 - tiltheta_old(0) - tiltheta_old(2)) * beta_spike_dens[j]/beta_slab_dens[j]);

      alpha_contrib = pstar/(tiltheta_old(1) + tiltheta_old(2)) - (1.0 - pstar)/(1.0 - tiltheta_old(1) - tiltheta_old(2));
      beta_contrib = qstar/(tiltheta_old(0) + tiltheta_old(2)) - (1.0 - qstar)/(1.0 - tiltheta_old(0) - tiltheta_old(2));
      
      // add derivative of log p(beta_j|theta) wrt to theta_01 to grad(0)
      grad(0) -= beta_contrib;
      // add derivative of log p(alpha_j|theta) wrt to theta_10 to grad(1)
      grad(1) -= alpha_contrib;
      // remember theta_11 appears in both log(p(alpha|theta)) and log(p(beta|theta))!
      // add derivative of log p(alpha_j | theta) + log p(beta_j | theta) to grad(2)
      grad(2) -= (alpha_contrib + beta_contrib);
      
      // now update the hessian!
      hess(0,0) += pow(beta_contrib, 2);
      hess(0,2) += pow(beta_contrib, 2);
      hess(1,1) += pow(alpha_contrib,2);
      hess(1,2) += pow(alpha_contrib,2);
      hess(2,2) += pow(alpha_contrib,2) + pow(beta_contrib,2);

    }
    
    // Remember to make hessian symmetric!
    hess(1,0) = hess(0,1);
    hess(2,0) = hess(0,2);
    hess(2,1) = hess(1,2);
    
    // Compute the full newton step
    v = -1.0 * arma::solve(hess, grad); // v = - hess^-1 * grad
    tgrad_v = arma::dot(grad, v); // tgrad_v = grad' * (-hess^-1) * grad
    step_size = 1.0;
    
    tiltheta_new = tiltheta_old + step_size * v;
    obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
    
    armijo_condition = true; // checks armijo condition: true iff obj_new < obj_old + 0.5 * step_size * tgrad_v
    in_unit_int = true; // does full newton step keep us within the interval
    
    if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5 ) ) in_unit_int = false; // abs(x - 0.5) > 0.5 iff x is outside [0,1]
    if(obj_new > obj_old + 0.5 * step_size * tgrad_v) armijo_condition = false;
    counter = 0;
    
    while( ((in_unit_int == false) || (armijo_condition == false)) && (counter < 200) ){
      step_size *= 0.5; // cut step size in half
      tiltheta_new = tiltheta_old + step_size * v;
      obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
    
      armijo_condition = true;
      in_unit_int = true;
      if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5 - eps) ) in_unit_int = false; // abs(x - 0.5) > 0.5 iff x is outside [0,1]
      if(obj_new > obj_old + 0.5 * step_size * tgrad_v) armijo_condition = false;
    }
    if( (counter == 200) && ( (in_unit_int == false) || (armijo_condition == false)) ){
      // After 200 back-tracking steps, we have not gotten sufficient decrease in the objective or are outside the unit interval
      Rcpp::Rcout << "[update_theta]: Backtracking unsuccessfully after 20 attempts!" << std::endl;
      if(in_unit_int == false){
        Rcpp::Rcout << "[update_theta]: tiltheta outside interval!" << std::endl;
        tiltheta_new.print();
      }
      if(armijo_condition == false){
        Rcpp::Rcout << "[update_theta]: objective did not decrease sufficiently" << std::endl;
      }
      Rcpp::stop("Terminating!");
    }
    
    diff = arma::norm(tiltheta_new - tiltheta_old);
    
    Rcpp::Rcout << "[update_theta]: Iter = " << iter << "  obj_old = " << obj_old << " obj_new = " << obj_new << " diff = " << diff << std::endl;
    //if(diff < eps) Rcpp::Rcout << "uh oh" << std::endl;

    iter++;
  } // closes main while loop of Newton step
  
  
  //tiltheta_new contains the updated values of theta_01, theta_10, theta_11
  theta(1) = tiltheta_new(0);
  theta(2) = tiltheta_new(1);
  theta(3) = tiltheta_new(2);
  theta(0) = 1.0 - arma::accu(tiltheta_new);
  
  Rcpp::Rcout << "gradient: " << std::endl;
  grad.print();
  
  Rcpp::Rcout << "Hessian: " << std::endl;
  hess.print();
  
  Rcpp::Rcout << " descent direction " << std::endl;
  v.print();
  
  
}

/*
// plain vanilla gradient descent
void update_theta_grad(arma::vec &theta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p, const int &max_iter, const double &eps)
{
  arma::vec grad = arma::zeros<arma::vec>(3);
  double grad_norm = 0.0;
  double step_size = 1.0;
  // initialize stuff needed to fill in gradient and hessian entries
  double pstar = 0.0;
  double qstar = 0.0;
  double alpha_contrib = 0.0; //alpha_contrib = [pstar/(theta_10 + theta_11) - (1 - pstar)/(1 - theta_10 - theta_11)]
  double beta_contrib = 0.0; //beta_contrib = [qstar/(theta_01 + theta_11) - (1 - qstar)/(1 - theta_01 - theta)11)]
  
  // inititalize stuff for the main Newton loop
  int iter = 0;
  int counter = 0;
  double diff = 1.0;
  bool armijo_condition = true; // Armijo condition: true if obj_new < obj_old + step_size * tgrad_v (condition in backtracking line search)
  bool in_unit_int = true; // checks that all entries of tiltheta are in [0,1]
  
  // tiltheta(0) = theta_01, tiltheta(1) = theta_10, tiltheta(2) = theta_11
  arma::vec tiltheta_new = arma::zeros<arma::vec>(3);
  tiltheta_new(0) = theta(1);
  tiltheta_new(1) = theta(2);
  tiltheta_new(2) = theta(3);
  double obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);

  
  arma::vec tiltheta_old = arma::zeros<arma::vec>(3);
  double obj_old = obj_new;

  while( (iter < max_iter) & (diff > eps) ){
    tiltheta_old = tiltheta_new;
    obj_old = obj_new;
    
    
    // Begin to compute gradient and hessian
    // We start with the gradient and hessian of log prior
    
    // derivative of log p(theta) wrt to theta_01
    grad(0) = -1.0 * delta(1)/tiltheta_old(0) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_01/theta_01 - delta_00/theta_00
                                                                                           // derivative of log p(theta) wrt to theta_10
    grad(1) = -1.0 * delta(2)/tiltheta_old(1) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_10/theta_10 - delta_00/theta_00
                                                                                           // derivative of log p(theta) wrt to theta_11
    grad(2) = -1.0 * delta(3)/tiltheta_old(2) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_11/theta_11 - delta_00/theta_00
    
    for(int j = 0; j < p; j++){
      
      pstar = 1.0/(1.0 + (tiltheta_old(1) + tiltheta_old(2))/(1.0 - tiltheta_old(1) - tiltheta_old(2)) * alpha_spike_dens[j]/alpha_slab_dens[j]);
      qstar = 1.0/(1.0 + (tiltheta_old(0) + tiltheta_old(2))/(1.0 - tiltheta_old(0) - tiltheta_old(2)) * beta_spike_dens[j]/beta_slab_dens[j]);
      
      alpha_contrib = pstar/(tiltheta_old(1) + tiltheta_old(2)) - (1.0 - pstar)/(1.0 - tiltheta_old(1) - tiltheta_old(2));
      beta_contrib = qstar/(tiltheta_old(0) + tiltheta_old(2)) - (1.0 - qstar)/(1.0 - tiltheta_old(0) - tiltheta_old(2));
      
      // add derivative of log p(beta_j|theta) wrt to theta_01 to grad(0)
      grad(0) -= beta_contrib;
      // add derivative of log p(alpha_j|theta) wrt to theta_10 to grad(1)
      grad(1) -= alpha_contrib;
      // remember theta_11 appears in both log(p(alpha|theta)) and log(p(beta|theta))!
      // add derivative of log p(alpha_j | theta) + log p(beta_j | theta) to grad(2)
      grad(2) -= (alpha_contrib + beta_contrib);
    }
    
    grad_norm = arma::norm(grad,2);
    step_size = 1.0;
    
    tiltheta_new = tiltheta_old - step_size * grad;
    
    in_unit_int = true;
    if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5) ) in_unit_int = false;
    counter = 0;
    while( (in_unit_int == false) && (counter < 200)){
      step_size *= 0.5;
      tiltheta_new = tiltheta_old - step_size * grad;
      in_unit_int = true;
      if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5 - eps) ) in_unit_int = false;
      counter++;
    }
    obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
    armijo_condition = true;
    in_unit_int = true;
    if(obj_new > obj_old - 0.5 * step_size * grad_norm) armijo_condition = false;
    if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5) ) in_unit_int = false;
    counter = 0;
    while( ((armijo_condition == false) || (in_unit_int == false)) && counter < 200){
      step_size *= 0.5;
      tiltheta_new = tiltheta_old - step_size * grad;
      obj_new = theta_objective(tiltheta_new, delta, alpha_spike_dens, alpha_slab_dens, beta_spike_dens, beta_slab_dens, p);
      armijo_condition = true;
      in_unit_int = true;
      if(obj_new > obj_old - 0.5 * step_size * grad_norm) armijo_condition = false;
      if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5) ) in_unit_int = false;
      counter++;
    }
    diff = arma::norm(tiltheta_new - tiltheta_old,2);
    Rcpp::Rcout << "[update_theta_grad]: Iter = " << iter << "  obj_old = " << obj_old << " obj_new = " << obj_new << " diff = " << diff <<  std::endl;
    iter++;
  }
  theta(1) = tiltheta_new(0);
  theta(2) = tiltheta_new(1);
  theta(3) = tiltheta_new(2);
  theta(0) = 1.0 - arma::accu(tiltheta_new);
}
*/
