//
//  update_theta.h
//    TO Do 7 May 2020: Instead of passing alpha and B, which are only used to compute pstar & qstar and the marginal densities
//    Introduce arguments alpha_spike_dens, alpha_slab_dens, B_spike_dens, B_slab_dens which are computed outsid.
//    It will make the code substantially easier to read (and avoids the xi1 * exp(-1.0 * xi1 * arma::norm(B.col(j))) terms )
//  Created by Sameer Deshpande on 5/1/20.
//

#ifndef _GUARD_update_theta_h
#define _GUARD_update_theta_h

#include<RcppArmadillo.h>
#include <stdio.h>



double theta_objective(const arma::vec &tiltheta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p);

void update_theta(arma::vec &theta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p, const int &max_iter, const double &eps);


//void update_theta_grad(arma::vec &theta, const arma::vec &delta, const std::vector<double> &alpha_spike_dens, const std::vector<double> &alpha_slab_dens, const std::vector<double> &beta_spike_dens, const std::vector<double> &beta_slab_dens, const int &p, const int &max_iter, const double &eps);


#endif /* update_theta_h */
