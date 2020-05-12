//
//  update_alpha.cpp
//  
//
//  Created by Sameer Deshpande on 5/1/20.
//

#include "update_alpha.h"

/*
 active shooting idea: first cycle over non-zero elements. upon convergence of the active set,
 cycle over remaining entries to check for violations of KKT condition. If there are violations, process is restarted
 */

void update_alpha(arma::vec &alpha, const arma::vec &R, const arma::mat &X, const double &sigma2, const double &lambda1, const double &lambda0, const arma::vec &theta, const int &max_iter, const double &eps, const int &n, const int &p, const bool &verbose)
{
  // initialize quantites used to update alpha
  double alpha_old = 0.0; // hold current value of $\alpha_{j}$
  double alpha_new = 0.0; // new value of $\alpha_{j}$
  double alpha_shift = 0.0; // defined as alpha_new - alpha_old
  
  // initialize quantities used to define penalties
  double pstar0 = 0.0; // $1/p_{j}^{\star}(0, \btheta)$,the probability that 0 is drawn from slab
  double pstar0_inv = 0.0; // 1/pstar0
  double pstar = 0.0; // $p_{j}^{\star}(\alpha_{j}, \btheta),$ the probability that current value of alpha_old is drawn from slab
  double lambda_star0 = 0.0; // will be lambda1 * pstar0 + lambda0 * (1.0 - pstar0)
  double lambda_star = 0.0; // equals lambda1 * pstar + lambda0 * (1.0 - pstar)
  
  // initialize quantities used to define the thresholds
  arma::vec resid = R - X * alpha; // partial residual, keeping B fixed
  double g = 0.0; // g(x) = (lambda_star(x) - lambda1) + 2n/sigma^2 * log(p^star(x)). g holds the value of g(0)
  double Delta = 0.0; // the refined threshold. We will set this to sigma2 * lambda_star0 when g = g(0) < 0

  // initialize quanties used to define arguments for our thresholding operators
  double z = 0.0; // $z = X_{j}^{\top}(R + X_{j}\alpha_{j}^{old})$
  double z_sgn = 0.0; // $\text{sign}(z).$ Will compute as (z > 0) - (z < 0)
  
  // initialize parameters for the main loop
  arma::vec active_set = arma::zeros<arma::vec>(p);
  for(int j = 0; j < p; j++){
    if(abs(alpha(j)) > eps) active_set(j) = 1;
    else active_set(j) = 0;
  }
  
  
  
  bool converged = true; // flag to indicate convergence in the active shooting algorithm
  bool violations = true; // flag to indicate violations of the KKT condition
  int iter = 0;
  
  
  // compute some quantities that will not change in the loop
  // Note: Entries of theta here are $\theta_{00}, \theta_{01}, \theta_{10}, \theta_{11}$
  pstar0_inv = 1.0 + (theta[0] + theta[1])/(theta[2] + theta[3]) * lambda0/lambda1;
  pstar0 = 1.0/pstar0_inv;
  lambda_star0 = lambda1 * (pstar0) + lambda0 * (1.0 - pstar0);
  
  if(lambda0 == lambda1){
    // when lambda0 == lambda1, this is just a Lasso problem and we can do iterative soft-thresholding
    
    while(iter < max_iter){
      while(iter < max_iter){ // inner loop is for active set convergence
        iter++;
        converged = true; // reset flag
        if(verbose == true) Rcpp::Rcout << "Iter " << iter;
        for(int j = 0; j < p; j++){
          if(active_set(j) == 1){
            alpha_old = alpha(j);
            z = alpha_old * ( (double) n) + arma::dot(X.col(j), resid); // Uses fact that $X_{j}^{\top}X_{j} = n$
            z_sgn = (z > 0.0) - (z < 0.0);
            alpha_new = 1.0/( (double) n) * std::max(abs(z) - sigma2 * lambda1, 0.0) * z_sgn; // soft thresholding at lambda1
            
            if(verbose == true){
              Rcpp::Rcout << "  [inner loop]: j = " << j;
              Rcpp::Rcout << "  z = " << z;
              Rcpp::Rcout << "  alpha_new = " << alpha_new << std::endl;
            }
            
            alpha_shift = alpha_new - alpha_old;
            
            alpha(j) = alpha_new;
            resid -= alpha_shift * X.col(j);
            if( abs(alpha_shift/alpha_old) > eps) converged = false; // if % change in alpha(j) > (100 * eps)%, no convergence
          } // closes if checking that index j is in active set
        } // closes first loop sweeing over indices j (attempts to update active set)
        
        // after this sweep over active set, converged = true means
        // all non-zero entries in alpha have changed by less than (100 * eps)%.
        // so we can break out of inner loop and go look for violations of KKT condition & condition
        if(converged == true) break;
      } // closes inner loop (for active set convergence)
      
      // we now loop over in-active set (active_set(j) = 0) and see if there are
      // any violations of KKT condition
      // if so, we update the violating alpha(j) and then move the index j to the active set
      violations = false; // reset flag checking for violations of KKT condition & condition
      for(int j = 0; j < p; j++){
        if(active_set(j) == 0){
          alpha_old = alpha(j);
          z = alpha_old * ( (double) n) + arma::dot(X.col(j), resid);
          z_sgn = (z > 0) - (z < 0);
          alpha_new = 1.0/( (double) n) * std::max(abs(z) - sigma2 * lambda1, 0.0) * z_sgn;
          if(verbose == true){
            Rcpp::Rcout << "  [inner loop]: j = " << j;
            Rcpp::Rcout << "  z = " << z;
            Rcpp::Rcout << "  alpha_new = " << alpha_new << std::endl;
          }
          //if(alpha_new != 0.0){
          // instead of checking for exact equality with 0,
          if(abs(alpha_new) > eps){
            // since j is in-active set, we expect alpha_j to remain 0.
            // so if alpha_new != 0 then we found a violation of KKT condition.
            // we will go ahead and update the value of alpha_j at this time
            violations = true;
            alpha_shift = alpha_new - alpha_old;
            alpha(j) = alpha_new;
            resid -= alpha_shift * X.col(j);
          } // closes if checking for violation of KKT condition & refined condition
        } // closes if checking that j is inactive
        
        // update active set.
        //if(alpha(j) != 0) active_set(j) = 1;
        if(abs(alpha(j)) > eps) active_set(j) = 1;
        else active_set(j) = 0;
      } // closes second sweep over indices j (looks for violations of KKT in in-active set)
      // after sweeping once over the in-active set, if we haven't found any violations, we can break out of outer loop
      if(violations == false) break;
    } // closes outer while loop
  } else{
    while(iter < max_iter){
      while(iter < max_iter){ // inner loop is for active set
        iter++;
        converged = true;
        if(verbose == true) Rcpp::Rcout << "Iter " << iter << std::endl;
        for(int j = 0; j < p; j++){
          if(active_set(j) == 1){
            alpha_old = alpha(j);
            z = alpha_old * ( (double) n) + arma::dot(X.col(j), resid); // Uses fact that $X_{j}^{\top}X_{j} = n$
            z_sgn = (z > 0.0) - (z < 0.0);
            
            // Compute pstar and lambda_star
            pstar = 1.0/(1.0 + (theta[0] + theta[1])/(theta[2] + theta[3]) * lambda0/lambda1 * exp(-1.0 * abs(alpha_old) * (lambda0 - lambda1)));
            lambda_star = lambda1 * pstar + lambda0 * (1.0 - pstar);
            
            // Which refined threshold to do we use?
            g = (lambda_star0 - lambda1) * (lambda_star0 - lambda1) + 2.0 * sigma2/ ( (double) n) * log(pstar0);
            if( (g > 0) && ( sqrt(sigma2) * (lambda0 - lambda1) > 2.0 * sqrt( (double) n) ) ){
              Delta = sigma2 * lambda1 + sqrt(2.0 * (double) n * log(pstar0_inv));
            } else Delta = sigma2 * lambda_star0;
            
            // Now carry out the refined thresholding
            if(abs(z) < Delta) alpha_new = 0.0;
            else alpha_new = 1.0/( (double) n) * z_sgn * std::max(abs(z) - sigma2 * lambda_star, 0.0);
            if(verbose == true){
              Rcpp::Rcout << "  [inner loop]: j = " << j;
              Rcpp::Rcout << "  z = " << z << "  Delta = " << Delta;
              Rcpp::Rcout << "  lambda_star = " << lambda_star;
              Rcpp::Rcout << "  alpha_new = " << alpha_new << std::endl;
            }
          
            alpha_shift = alpha_new - alpha_old;
            alpha(j) = alpha_new;
            resid -= alpha_shift * X.col(j);
            if( abs(alpha_shift/alpha_old) > eps) converged = false; // if % change in alpha(j) > (100 * eps)%, no convergence
          } // closes if checking that index j is in active set
        } // closes first loop over indices j (only updates alpha[j] for j in active set)
        

        
        // after the sweep over active set, converged = true means
        // all non-zero entries in alpha have changed by less than (100 * eps)%
        // so we break out of inner loop and go look for violations of KKT condition & refined condition
        if(converged == true) break;
      } // closes inner loop (for active set convergence)

      // we now loop over in-active set (active_set(j) = 0) and see if there are
      // any violations of KKT condition
      // if so, we update the violating alpha(j) and then move the index j to the active set
      violations = false;
      for(int j = 0; j < p; j++){
        if(active_set(j) == 0){
          alpha_old = alpha(j);
          z = alpha_old * ( (double) n) + arma::dot(X.col(j), resid); // Uses fact that $X_{j}^{\top}X_{j} = n$
          z_sgn = (z > 0.0) - (z < 0.0);
          
          // Compute pstar and lambda_star
          pstar = 1.0/(1.0 + (theta[0] + theta[1])/(theta[2] + theta[3]) * lambda0/lambda1 * exp(-1.0 * abs(alpha_old) * (lambda0 - lambda1)));
          lambda_star = lambda1 * pstar + lambda0 * (1.0 - pstar);
          
          // Which refined threshold to do we use?
          g = (lambda_star0 - lambda1) * (lambda_star0 - lambda1) + 2.0 * sigma2/ ( (double) n) * log(pstar0);
          if( (g > 0) && ( sqrt(sigma2) * (lambda0 - lambda1) > 2.0 * sqrt( (double) n) ) ){
            Delta = sigma2 * lambda1 + sqrt(2.0 * (double) n * log(pstar0_inv));
          } else Delta = sigma2 * lambda_star0;
          
          // Now carry out the refined thresholding
          if(abs(z) < Delta) alpha_new = 0.0;
          else alpha_new = 1.0/( (double) n) * z_sgn * std::max(abs(z) - sigma2 * lambda_star, 0.0);
          
          if(verbose == true){
            Rcpp::Rcout << "  [outer loop]: j = " << j;
            Rcpp::Rcout << "  z = " << z << "  Delta = " << Delta;
            Rcpp::Rcout << "  lambda_star = " << lambda_star;
            Rcpp::Rcout << "  alpha_new = " << alpha_new << std::endl;
          }
          
          //if(alpha_new != 0.0){
          if(abs(alpha_new) > eps){
            // since j is in-active set, we expect alpha_j to remain 0.
            // so if alpha_new != 0 then we found a violation of KKT condition.
            // we will go ahead and update the value of alpha_j at this time
            violations = true;
            alpha_shift = alpha_new - alpha_old;
            alpha(j) = alpha_new;
            resid -= alpha_shift * X.col(j);
          } // closes if checking if there's a violation of the KKT condition & refined condition
        } // closes if checking that index j is in in-active set
        
        // now we are in a position to update the active set
        //if(alpha(j) != 0) active_set(j) = 1;
        if(abs(alpha(j)) > eps) active_set(j) = 1;
        else active_set(j) = 0;
      } // closes second sweep over indices j (looks for violations of KKT & refined condition in in-active set)
      // after sweeping over the in-active set, if we haven't found any violations, we can break out of outer loop
      if(violations == false) break;
    }
  } // closes else for lambda0 != lambda1
  if( (iter == max_iter) && ( (converged == false) || (violations == true) ) ){
    // reached maximum number of iterations and either (i) did not converge over active set or (ii) found violations of refined characterization
  }

  
}
