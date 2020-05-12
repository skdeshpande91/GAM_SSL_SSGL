//
//  update_alpha.h
//  
//
//  Created by Sameer Deshpande on 5/1/20.
//

#ifndef _GUARD_update_alpha_h
#define _GUARD_update_alpha_h

#include<RcppArmadillo.h>
#include <math.h>
#include <stdio.h>

void update_alpha(arma::vec &alpha, const arma::vec &R, const arma::mat &X, const double &sigma2, const double &lambda1, const double &lambda0, const arma::vec &theta, const int &max_iter, const double &eps, const int &n, const int &p, const bool &verbose);

#endif /* update_alpha_h */
