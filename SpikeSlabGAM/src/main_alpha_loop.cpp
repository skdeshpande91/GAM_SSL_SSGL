arma::vec beta_new(p);

while(iter < max_iter){
  while(iter < max_iter){ // inner loop is for active set convergence
    iter++;
    converged = true; //reset convergence flag
    
    for(int j = 0; j < p; j++){
      if(active_set[j] == 1){
        beta_old = B.col(j);
        z = (double) n * beta_old + Phi.slice(j).t() * resid; // need to double check syntax here
        z_norm = arma::norm(z,2);
        if(z_norm < sigma2 * lambda1) beta_new.fill(0.0);
        else beta_new = 1.0/( (double) n) * max(1.0 - sigma2 * lambda1/z_norm, 0.0) * z;
        
        beta_shift = beta_new - beta_old;
        resid -= Phi.slice(j) * beta_shift;
        B.col(j) = beta_new;
        
      }
    }
  }
}






void update_theta(arma::vec &theta, const arma::vec &delta, const arma::vec &alpha, const arma::mat &B, const double &lambda1, const double &lambda0, const double &xi1, const double &xi0,  const int &max_iter, const double &eps, const int &p, )
{
  
  
  arma::vec grad_full = arma::zeros<vec>(3);
  arma::mat hess_full = arma::zeros<mat>(3,3);
  arma::vec search_dif = arma::zeros<mat>(3);
  
  double pstar = 0.0;
  double qstar = 0.0;
  
  double alpha_contrib = 0.0;
  double beta_contrib = 0.0;
  
  
  // stuff for iteration
  int iter = 0;
  double diff = 1.0; // different in value of theta
  
  // let tiltheta be the set of 3 free parameters: theta_01, theta_10, theta_11
  // ie. tiltheta = theta(1:3)
  arma::vec tiltheta_old = arma::zeros<vec>(3);
  arma::vec tiltheta_new = arma::zeros<vec>(3)
  
  // stuff for newton step & backtracking line search
  arma::vec v = arma::zeros<vec>(3);
  double tgrad_v = 0.0;
  double obj_new = 0.0;
  double obj_old = 0.0;
  bool suff_decrease = true;
  bool in_unit_int = true;
  double step_size = 1.0;
  
  if( (lambda1 == lambda0) && (xi1 == xi0)){
    theta(0) = eps/3.0;
    theta(1) = eps/3.0;
    theta(2) = eps/3.0;
    theta(3) = 1 - eps;
    // when both spike and slab are equal to 0, set theta = (eps/3, eps/3, eps/3, 1-eps) so that everything is in slab
  } else{
    
    // begin main newton step loop
    while( (iter < max_iter) & (diff > eps) ){
      
      tiltheta_old = tiltheta_new;
      obj_old = obj_new;
      
      
      // derivative of log p(theta) wrt to theta_01
      grad(0) = -1.0 delta(1)/tiltheta_old(0) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_01/theta_01 - delta_00/theta_00
      
      // derivative of log p(theta) wrt to theta_10
      grad(1) = -1.0 * delta(2)/tiltheta_old(1) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_10/theta_10 - delta_00/theta_00
      
      // derivative of log p(theta) wrt to theta_11
      grad(2) = -1.0 * delta(3)/tiltheta_old(2) + delta(0)/(1.0 - arma::accu(tiltheta_old)); // delta_11/theta_11 - delta_00/theta_00
      
      
      // initialize elements of hessian with the hessian of the log prior
      //remember tiltheta_*(0) = theta_01, tiltheta_*(1) = theta_10, tiltheta_*(2) = theta_11
      // which is why the indices between hess & tiltheta_* differ from the indices in delta by 1
      
      hess(0,0) = delta(1)/pow(tiltheta_old(0),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old), 2);
      hess(0,1) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
      hess(0,2) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
      hess(1,1) = delta(2)/pow(tiltheta_old(1),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
      hess(1,2) = delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
      hess(2,2) = delta(3)/pow(tiltheta_old(2),2) + delta(0)/pow(1.0 - arma::accu(tiltheta_old),2);
      
      for(int j = 0; j < p; j++){
        // compute pstar * qstar
        // remember tiltheta = (theta_01, theta_10, theta_11)
        pstar = 1.0/(1.0 + (tiltheta_old(1) + tiltheta_old(2))/(1.0 - tiltheta_old(1) - tiltheta_old(2)) * lambda0/lambda1 * exp(-1.0 * abs(alpha(j)) * (lambda0 - lambda1)));
        qstar = 1.0/(1.0 + (tiltheta_old(0) + tiltheta_old(2))/(1.0 - tiltheta_old(0) - tiltheta_old(2))) * xi0/xi1 * exp(-1.0 * abs(arma::norm(B.col(j), 2)) * (xi0 - xi1));
        
        
        // alpha_contrib = [pstar/(theta_10 + theta_11) - (1 - pstar)/(1 - theta_10 - theta_11)]
        // beta_contrib = [qstar/(theta_01 + theta_11) - (1 - qstar)/(1 - theta_01 - theta)11)]
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
      } // closes loop over covariates j to compute gradient and hessian
      
      // now fill in the lower triangle of hess
      hess(1,0) = hess(0,1);
      hess(2,0) = hess(0,2);
      hess(2,1) = hess(1,2);
      
      // at this point we have everything we need to carry out the newton step
      // first we compute the direction p = hess^-1 * grad
      v = -1.0 * arma::solve(hess, grad); // v = - hess^-1 * grad
      tgrad_v = arma::dot(grad, v); // tgrad_v = grad' * (-hess^-1) * grad
      step_size = 1.0;
      
      tiltheta_new = theta_old + step_size * v;
      obj_new = theta_objective(tiltheta_new, delta, alpha, B, lambda1, lambda0, xi1, xi0);
      
      step_size = 1.0;
      tiltheta_new = theta_old - step_size * search_dir;
      
      obj_new = theta_objective(tiltheta_new, delta, alpha, B, lambda1, lambda0, xi1, xi0);
      
      suff_decrease = true; // does full newton step sufficient reduce objective
      in_unit_int = true; // does full newton step keep us within the interval
      
      if( arma::any(arma::abs(tiltheta_new - 0.5) > 0.5 ) ) in_unit_int = false;
      if(obj_new > obj_old + 0.5 * tgrad_v) suff_decrease = false;
      counter = 0;
      while( ((in_unit_int == false) || (suff_decrease == false)) && (counter < 10) ){
        step_size *= 0.5; // cut step size in half
        tiltheta_new = theta_old - step_size * search_dir;
        obj_new = theta_objective(tiltheta_new, delta, alpha, B, lambda1, lambda0, xi1, xi0);
        suff_decrease = true; // does full newton step sufficient reduce objective
        in_unit_int = true; // does full newton step keep us within the interval
        counter++;
      }
      if( (counter == 10) && ((in_unit_int == false) || (suff_decrease == false)) ){
        Rcpp::Rcout << "10 backtracking steps didn't suffice!" << std::endl;
      }
      diff = step_size * arma::norm(search_dir);
      iter++;
    }
    
    
  }
  
  
  
  
}
