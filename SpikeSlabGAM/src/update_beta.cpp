//
//  update_beta.cpp
//  
//
//  Created by Sameer Deshpande on 5/1/20.
//

#include "update_beta.h"


void update_beta(arma::mat &B, const arma::vec &R, const arma::cube &Phi, const double &sigma2, const double &xi1, const double &xi0, const arma::vec &theta, const int &max_iter, const double &eps, const int &n, const int &p, const int &D, const bool verbose)
{
  // initialize quantities used to update $\bbeta_{j}$
  arma::vec beta_old = arma::zeros<arma::vec>(D); // holds current value of $\bbeta_{j}$: B.col(j)
  arma::vec beta_new = arma::zeros<arma::vec>(D); // holds new value of $\bbeta_{j}$
  arma::vec beta_shift = arma::zeros<arma::vec>(D); // defined as beta_new - beta_old
  
  arma::mat B_old = B;
  double diff = 0;
  // initialize quantities used to define penalties
  double qstar0 = 0.0; // $1/q_{j}^{\star}(0, \btheta)$,the probability that vector of 0's is drawn from slab
  double qstar0_inv = 0.0; // 1/qstar0
  double qstar = 0.0; // $q_{j}^{\star}(\bbeta)$, the probability that current value of $\bbeta_{j}$ (beta_old) is drawn from slab
  double xi_star0 = 0.0; // equals  xi1 * qstar0 + xi0 * (1.0 - qstar0)
  double xi_star = 0.0; // equals xi1 * qstar + xi0 * (1.0 - qstar)
  
  // initialize quantities used to define thresholds
  arma::vec resid = R;
  for(int j = 0; j < p; j++) resid -= Phi.slice(j) * B.col(j);
  double h = 0.0; // h(0) where $h(\bbeta) = (\xi^{\star}(\bbeta, \btheta) - \xi_{1})^{2} + \frac{2n}{\sigma^{2}}\log{q^{\star}(\bbeta, \btheta)}$$
  double Delta = 0.0; // the refined threshold. We will set this to sigma2 * xi1 + sqrt(2.0 * n * sigma2 * log(qstar0_inv))
  
  // initialize quantities used to define arguments for thresholding operations
  arma::vec z = arma::zeros<arma::vec>(D); //$z = \Phi_{j}^{\top}(resid + \Phi_{j}\bbeta_{j}) = n * beta_old + \Phi_{j}^{\top} resid
  double z_norm = 0.0; // norm of z
  
  // initialize parameters for the main loop
  //std::vector<int> active_set(p);
  arma::vec active_set = arma::zeros<arma::mat>(p);
  for(int j = 0; j < p; j++){
    if(arma::norm(B.col(j),2) > eps) active_set(j) = 1;
    else active_set(j) = 0;
  }
  
  int iter = 0;
  bool converged = true; // flag to indicate convergence in the active set
  bool violations = true;
  
  
  // compute some quantities that will not change in the loop
  // Note entries in theta here are: $\theta_{00}, \theta_{01}, \theta_{10}, \theta_{11}$
  qstar0_inv = 1.0 + (theta[0] + theta[2])/(theta[1] + theta[4]) * pow(xi0/xi1, D);
  qstar0 = 1.0/qstar0_inv;
  xi_star0 = xi1 * qstar0 + xi0 * (1.0 - qstar0);
  
  if(xi0 == xi1){
    // when xi0 == xi1, this is just a group lasso problem and we don't compute any of the refined thresholds
    while(iter < max_iter){
      while(iter < max_iter){ // inner loop is for active set convergence
        iter++;
        converged = true; // reset convergence flag
        
        if(verbose == true) Rcpp::Rcout << "Iter = " << iter << std::endl;
        B_old = B;
        for(int j = 0; j < p; j++){
          if( (active_set(j) == 1) && (arma::norm(B.col(j),2) > eps) ){
            beta_old = B.col(j);
            z = (double) n * beta_old + Phi.slice(j).t() * resid; // need to check syntax here
            z_norm = arma::norm(z,2);
            if(z_norm <= sigma2 * xi1) beta_new.fill(0.0);
            else{
              beta_new = z;
              beta_new *= 1.0/( (double) n) * (1.0 - sigma2 * xi1/z_norm);
            }
            
            if(verbose == true){
              Rcpp::Rcout <<"[inner loop]: j = " << j;
              Rcpp::Rcout << "  z_norm = " << z_norm;
              Rcpp::Rcout << "  beta_new_norm = " << arma::norm(beta_new,2) << std::endl;
            }
            
            beta_shift = beta_new - beta_old;
            B.col(j) = beta_new;
            resid -= Phi.slice(j) * beta_shift;
            //if(arma::norm(beta_shift) > eps) converged = false;
          } // closes if checking that index j is in active set (i.e. B.col(j) != 0)
        } // closes first loop sweeping over indices j (attempts to update active set)
        
        // now that we ahve finished sweeping over indices j, let's compute the difference in B
        diff = arma::norm(B - B_old, 2);
        if(verbose == true) Rcpp::Rcout << "[inner loop]: diff = " << diff << std::endl;
        if(diff > eps) converged = false;
        else converged = true;
        // after this sweep over active set, converged = true means
        // all columns of B have changed by less than eps (in euclidean norm)
        // so we can break out of inner loop and go look for violations of KKT condition
        if(converged == true) break;
      } // closes inner loop (for active set convergence)
      
      // we now loop over in-active set (active_set(j) = 0) and see if there are
      // any violations of KKT condition
      // if so, we update the violating alpha(j) and then move the index j to the active set
      violations = false; // reset flag checking for violations of KKT condition
      
      for(int j = 0; j < p; j++){
        if( (active_set(j) == 0) || (arma::norm(B.col(j),2) < eps)){
          beta_old = B.col(j);
          z = (double) n * beta_old + Phi.slice(j).t() * resid; // need to check syntax here
          z_norm = arma::norm(z,2);
          if(z_norm <= sigma2 * xi1) beta_new.fill(0.0);
          else{
            beta_new = z;
            beta_new *= 1.0/( (double) n) * (1.0 - sigma2 * xi1/z_norm);
          }
          
          if(verbose == true){
            Rcpp::Rcout <<"[outer loop]: j = " << j;
            Rcpp::Rcout << "  z_norm = " << z_norm;
            Rcpp::Rcout << "  beta_new_norm = " << arma::norm(beta_new,2) << std::endl;
          }
          
          if(arma::norm(beta_new,2) > eps){
            // since j is in in-active set, we expect $\bbeta_j$ to remain 0
            // if it is not exactly 0, then we found a violation of KKT condition
            // we will go ahead and update $\bbeta_j$ at this time
            violations = true;
            beta_shift = beta_new - beta_old;
            B.col(j) = beta_new;
            resid -= Phi.slice(j) * beta_shift;
          }
         
          
          
        } // closes if checking that j is in in-active set
        
        // update active set
        if(arma::norm(B.col(j), 2) != 0.0) active_set(j) = 1;
        else active_set(j) = 0;
      } // closes second loop sweeping over indices j (looks for violation of KKT)
      
      // after sweeping over indices j (looking for violations of KKT in in the active set)
      // if we haven't found any violations, we can break out of the outer loop
      if(violations == false) break;
    } // closes outer loop
  } else{
    while(iter < max_iter){
      while(iter < max_iter){ // inner loop is for active set convergence
        iter++;
        converged = true; // reset convergence flag
        B_old = B;
        diff = 1.0;
        if(verbose == true) Rcpp::Rcout << "Iter = " << iter << " initially active set size = " << arma::accu(active_set) << std::endl;
        for(int j = 0; j < p; j++){
          if( (active_set(j) == 1) && (arma::norm(B.col(j),2) > eps) ){
            beta_old = B.col(j);
            z = (double) n * beta_old + Phi.slice(j).t() * resid; // need to check syntax here
            z_norm = arma::norm(z,2);
            
            
            // compute qstar and xi_star
            qstar = 1.0/(1.0 + (theta[0] + theta[2])/(theta[1] + theta[4]) * pow(xi0/xi1,D) * exp(-1.0 * arma::norm(beta_old,2) * (xi0 - xi1)));
            xi_star = xi1 * qstar + xi0 * (1.0 - qstar);
            
            // Determine which threshold to use
            h = (xi_star0 - xi1) * (xi_star0 - xi1) + 2.0 * (double) n / sigma2 * log(qstar0);
            if( (h > 0) && ( sqrt(sigma2) * (xi0 - xi1) > 2.0 * sqrt( (double) n) ) ){
              Delta = sigma2 * xi1 + sqrt(2.0 * (double) n * log(qstar0_inv));
            } else{
              Delta = sigma2 * xi_star0;
            }
          
            if(z_norm <= Delta) beta_new.fill(0.0);
            else if(z_norm <= sigma2 * xi_star) beta_new.fill(0.0);
            else{
              beta_new = z;
              beta_new *= 1.0/( (double) n ) * (1.0 - sigma2 * xi_star/z_norm);
            }
            
            beta_shift = beta_new - beta_old;
            if(verbose == true){
              Rcpp::Rcout <<"[inner loop]: j = " << j;
              Rcpp::Rcout << "  z_norm = " << z_norm;
              Rcpp::Rcout << "  Delta = " << Delta;
              Rcpp::Rcout << "  xi_star = " << xi_star;
              Rcpp::Rcout << "  beta_new_norm = " << arma::norm(beta_new,2) << std::endl;
            }
            B.col(j) = beta_new;
            resid -= Phi.slice(j) * beta_shift;
            //if(arma::norm(beta_shift,2) > eps) converged = false;
          } // closes if checking that index j is in active set (i.e. B.col(j) != 0)
        } // closes first loop sweeping over indices j (attempts to update active set)
        
        
        diff = arma::norm(B - B_old,2);
        if(verbose == true) Rcpp::Rcout << "[inner loop]: diff = " << diff << std::endl;
        if(diff < eps)
        
        
        // after this sweep over active set, converged = true means
        // all columns of B have changed by less than eps (in euclidean norm)
        // so we can break out of inner loop and go look for violations of KKT condition
        if(converged == true) break;
      } // closes inner loop (for active set convergence)
      
      // we now loop over in-active set (active_set(j) = 0) and see if there are
      // any violations of KKT condition
      // if so, we update the violating alpha(j) and then move the index j to the active set
      violations = false; // reset flag checking for violations of KKT condition
      
      for(int j = 0; j < p; j++){
        if( (active_set(j) == 0) || (arma::norm(B.col(j),2) < eps) ){
          beta_old = B.col(j);
          z = (double) n * beta_old + Phi.slice(j).t() * resid; // need to check syntax here
          z_norm = arma::norm(z,2);
          // compute qstar and xi_star
          qstar = 1.0/(1.0 + (theta[0] + theta[2])/(theta[1] + theta[4]) * pow(xi0/xi1,D) * exp(-1.0 * arma::norm(beta_old,2) * (xi0 - xi1)));
          xi_star = xi1 * qstar + xi0 * (1.0 - qstar);
          
          // Determine which threshold to use
          h = (xi_star0 - xi1) * (xi_star0 - xi1) + 2.0 * (double) n / sigma2 * log(qstar0);
          if( (h > 0) && ( sqrt(sigma2) * (xi0 - xi1) > 2.0 * sqrt( (double) n) ) ){
            Delta = sigma2 * xi1 + sqrt(2.0 * (double) n * log(qstar0_inv));
          } else{
            Delta = sigma2 * xi_star0;
          }
          
          if(z_norm <= Delta) beta_new.fill(0.0);
          else if(z_norm <= sigma2 * xi_star) beta_new.fill(0.0);
          else{
            beta_new = z;
            beta_new *= 1.0/( (double) n ) * (1.0 - sigma2 * xi_star/z_norm);
          }
          
          if(verbose == true){
            Rcpp::Rcout <<"[outer loop]: j = " << j;
            Rcpp::Rcout << "  z_norm = " << z_norm;
            Rcpp::Rcout << "  Delta = " << Delta;
            Rcpp::Rcout << "  xi_star = " << xi_star;
            Rcpp::Rcout << "  beta_new_norm = " << arma::norm(beta_new,2) << std::endl;
          }
          
          if(arma::norm(beta_new,2) > eps){
            // since j is in in-active set, we expect $\bbeta_j$ to remain 0
            // if it is not exactly 0, then we found a violation of KKT condition
            // we will go ahead and update $\bbeta_j$ at this time
            violations = true;
            beta_shift = beta_new - beta_old;
            B.col(j) = beta_new;
            resid -= Phi.slice(j) * beta_shift;
          }
        } // closes if checking that j is in in-active set
        
        // update active set
        //if(arma::norm(B.col(j), 2) != 0.0) active_set(j) = 1;
        if(arma::norm(B.col(j)) > eps) active_set(j) = 1;
        else active_set(j) = 0;
      } // closes second loop sweeping over indices j (looks for violation of KKT)
      if(verbose == true) Rcpp::Rcout << "Iter = " << iter << "after outer loop, active set size =  " << arma::accu(active_set) << std::endl;
      // after sweeping over indices j (looking for violations of KKT in in the active set)
      // if we haven't found any violations, we can break out of the outer loop
      if(violations == false) break;
    } // closes outer loop
  } // closes if/else checking whether xi0 == xi1
  
  if(iter == max_iter) Rcpp::Rcout << "[update_B]: Hit max iter!" << std::endl;
  
  
  
}
