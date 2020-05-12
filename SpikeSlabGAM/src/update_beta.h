//
//  update_beta.h
//  
//
//  Created by Sameer Deshpande on 5/1/20.
//

#ifndef _GUARD_update_beta_h
#define _GUARD_update_beta_h

#include<RcppArmadillo.h>
#include<math.h>
#include <stdio.h>

void update_beta(arma::mat &B, const arma::vec &R, const arma::cube &Phi, const double &sigma2, const double &xi1, const double &xi0, const arma::vec &theta, const int &max_iter, const double &eps, const int &n, const int &p, const int &D, const bool verbose);



#endif /* update_beta_h */
