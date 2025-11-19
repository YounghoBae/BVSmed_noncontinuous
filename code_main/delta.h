#ifndef __DELTA_H__
#define __DELTA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::depends(RcppEigen)]]

// Library Functions
MatrixXd add_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd delete_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
MatrixXd swap_step2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X);
double U(const VectorXd& delta, const VectorXd& y, const MatrixXd& M, const MatrixXd& X, const VectorXd& treat, const VectorXd& alpha0, const VectorXd& alpha, double alpha_p, const VectorXd& psi);
VectorXd grad_U(const VectorXd& delta, const VectorXd& y, const MatrixXd& M, const MatrixXd& X, const VectorXd& treat, const VectorXd& alpha0, const VectorXd& alpha, double alpha_p, const VectorXd& psi);
MatrixXd HMC(MatrixXd eff_coeff, const VectorXd& y, const MatrixXd& M, const MatrixXd& X, const VectorXd& treat, const VectorXd& alpha0, const VectorXd& alpha, double alpha_p, const VectorXd& psi, double epsilon, double L);
  
MatrixXd update_delta2(MatrixXd eff_coeff, const double V_delta, const VectorXd& psi, const double theta_omega, const VectorXd& y, const VectorXd& alpha0, const MatrixXd& M, const VectorXd& alpha, const double alpha_p, const VectorXd& treat, const MatrixXd& X, double epsilon, double L);

// Functions Define
// 1-1. add update
MatrixXd add_step2(MatrixXd eff_coeff,
                   const double V_delta,
                   const VectorXd& psi,
                   const double theta_omega,
                   const VectorXd& y,
                   const VectorXd& alpha0,
                   const MatrixXd& M,
                   const VectorXd& alpha,
                   const double alpha_p,
                   const VectorXd& treat,
                   const MatrixXd& X){
  int q = M.cols();
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int l_idx = random_zero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  
  // add prior ratio
  double log_prior_part1 = log_normal_dist(delta_star(l_idx), pow(psi(l_idx), 2));
  double log_prior_ratio = log_prior_part1 + log(theta_omega/(1.0-theta_omega));
  
  // add proposal ratio
  double log_proposal_part1 = -log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log((q-g_omega)/(g_omega+1.0));
  
  // add likelihood ratio
  double log_like1 = loglike_binary(y, delta_star, M, X, treat, alpha0, alpha, alpha_p);
  double log_like2 = loglike_binary(y, delta,      M, X, treat, alpha0, alpha, alpha_p);
  
  double log_likelihood_ratio = log_like1 - log_like2;
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 2-1. delete update
MatrixXd delete_step2(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X){
  int q = M.cols();
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  double g_omega = omega.sum();
  int m_idx = random_nonzero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(m_idx) = 0.0;
  
  // delete prior ratio
  double log_prior_part1 = -log_normal_dist(delta(m_idx), pow(psi(m_idx), 2));
  double log_prior_ratio = log_prior_part1 + log((1.0-theta_omega)/theta_omega);
  
  // delete proposal ratio
  double log_proposal_part1 = log_normal_dist(delta(m_idx), V_delta);
  double log_proposal_ratio = log_proposal_part1 + log(g_omega/(q-g_omega+1.0));
  
  // delete likelihood ratio
  double log_like1 = loglike_binary(y, delta_star, M, X, treat, alpha0, alpha, alpha_p);
  double log_like2 = loglike_binary(y, delta,      M, X, treat, alpha0, alpha, alpha_p);
  
  double log_likelihood_ratio = log_like1 - log_like2;
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// 3-1. swap update
MatrixXd swap_step2(MatrixXd eff_coeff,
                    const double V_delta,
                    const VectorXd& psi,
                    const double theta_omega,
                    const VectorXd& y,
                    const VectorXd& alpha0,
                    const MatrixXd& M,
                    const VectorXd& alpha,
                    const double alpha_p,
                    const VectorXd& treat,
                    const MatrixXd& X){
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  int l_idx = random_zero_index_given(omega, gamma);
  int m_idx = random_nonzero_index_given(omega, gamma);
  
  VectorXd delta_star = delta;
  delta_star(l_idx) = Rcpp::rnorm(1, delta(l_idx), sqrt(V_delta))(0);
  delta_star(m_idx) = 0.0;
  VectorXd omega_star = omega;
  omega_star(l_idx) = 1.0;
  omega_star(m_idx) = 0.0;
  
  // swap prior ratio
  double num_part_prior = log_normal_dist(delta_star(l_idx), pow(psi(l_idx), 2));
  double den_part_prior = log_normal_dist(delta(m_idx), pow(psi(m_idx), 2));
  double log_prior_ratio = num_part_prior - den_part_prior;
  
  // swap proposal ratio
  double num_part_proposal = log_normal_dist(delta(m_idx), V_delta);
  double den_part_proposal = log_normal_dist(delta_star(l_idx), V_delta);
  double log_proposal_ratio = num_part_proposal - den_part_proposal;
  
  // swap likelihood ratio
  double log_like1 = loglike_binary(y, delta_star, M, X, treat, alpha0, alpha, alpha_p);
  double log_like2 = loglike_binary(y, delta,      M, X, treat, alpha0, alpha, alpha_p);
  
  double log_likelihood_ratio = log_like1 - log_like2;
  
  // add accept/reject
  double log_accept_rate_delta = log_prior_ratio + log_proposal_ratio + log_likelihood_ratio;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(log_accept_rate_delta > log_u){
    eff_coeff.col(2) = delta_star;
    eff_coeff.col(3) = omega_star;
  }
  return eff_coeff;
}

// U function = -log p(delta)
double U(const VectorXd& delta,
         const VectorXd& y,
         const MatrixXd& M,
         const MatrixXd& X,
         const VectorXd& treat,
         const VectorXd& alpha0,
         const VectorXd& alpha,
         double alpha_p,
         const VectorXd& psi){
  
  VectorXd A = alpha0 + M * delta + X * alpha + alpha_p * treat;
  
  double part1 = y.transpose() * A;
  
  VectorXd part2_sub = VectorXd::Zero(A.rows());
  for(int i=0; i<A.rows(); i++){
    part2_sub(i) = log(1 + exp(A(i)));
  }
  double part2 = part2_sub.sum();
  
  VectorXd part3_sub = VectorXd::Zero(delta.rows());
  for(int j=0; j<delta.rows(); j++){
    part3_sub(j) = (delta(j) * delta(j)) / (psi(j) * psi(j));
  }
  double part3 = 0.5 * part3_sub.sum();
  
  double out = - part1 + part2 + part3;
  
  return out;
}

VectorXd grad_U(const VectorXd& delta,
                const VectorXd& y,
                const MatrixXd& M,
                const MatrixXd& X,
                const VectorXd& treat,
                const VectorXd& alpha0,
                const VectorXd& alpha,
                double alpha_p,
                const VectorXd& psi){
  
  VectorXd A = alpha0 + M * delta + X * alpha + alpha_p * treat;
  
  VectorXd A2 = VectorXd::Zero(A.rows());
  for(int i=0; i<A.rows(); i++){
    A2(i) = exp(A(i)) / (1 + exp(A(i)));
  }
  
  VectorXd del_psi_vec = VectorXd::Zero(psi.rows());
  for(int j=0; j<psi.rows(); j++){
    del_psi_vec(j) = delta(j) / (psi(j) * psi(j));
  }
  
  VectorXd out = - M.transpose() * y + M.transpose() * A2 + del_psi_vec;
  
  return out;
}

// HMC function
MatrixXd HMC(MatrixXd eff_coeff,
             const VectorXd& y,
             const MatrixXd& M,
             const MatrixXd& X,
             const VectorXd& treat,
             const VectorXd& alpha0,
             const VectorXd& alpha,
             double alpha_p,
             const VectorXd& psi,
             double epsilon,
             double L){
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  VectorXd update_idx = nonzero_index(omega);
  int update_size = update_idx.rows();
  
  if(update_size > 0){
    MatrixXd new_M     = MatrixXd::Zero(M.rows(), update_size);
    VectorXd new_delta = VectorXd::Zero(update_size);
    VectorXd new_psi   = VectorXd::Zero(update_size);
    
    for(int i=0; i<update_size; i++){
      int idx = update_idx(i);
      new_M.col(i) = M.col(idx);
      new_delta(i) = delta(idx);
      new_psi(i)   = psi(idx);
    }
    
    // Current delta
    VectorXd proposed_delta = new_delta;
    
    // Sample a momentum and initialize
    VectorXd p = generateNormalSamples(proposed_delta.rows(), 0, 1);
    VectorXd current_p = p;
    
    // Make the first half step
    p = p - epsilon/2 * grad_U(proposed_delta, y, new_M, X, treat, alpha0, alpha, alpha_p, new_psi);
      
    // Do the leapfrog part
    for(int j=0; j<L; j++){
      proposed_delta = proposed_delta + epsilon * p;
      
      if(j < L){
        p = p - epsilon * grad_U(proposed_delta, y, new_M, X, treat, alpha0, alpha, alpha_p, new_psi);
      }else{
        p = p - epsilon * grad_U(proposed_delta, y, new_M, X, treat, alpha0, alpha, alpha_p, new_psi) * 0.5;
      }
    }
      
    // Accept or reject the proposal (q)
    double current_U  = U(new_delta,      y, new_M, X, treat, alpha0, alpha, alpha_p, new_psi);
    double proposed_U = U(proposed_delta, y, new_M, X, treat, alpha0, alpha, alpha_p, new_psi);
    
    VectorXd current_K_vec = VectorXd::Zero(p.rows());
    VectorXd proposed_K_vec = VectorXd::Zero(p.rows());
    
    for(int k=0; k<p.rows(); k++){
      current_K_vec(k) = current_p(k) * current_p(k);
      proposed_K_vec(k) = p(k) * p(k);
    }
    
    double current_K  = 0.5 * current_K_vec.sum();
    double proposed_K = 0.5 * proposed_K_vec.sum();
    
    double log_u = log(Rcpp::runif(1)(0));
    
    if (log_u < (-(proposed_U+proposed_K)+(current_U+current_K))){
      for(int k=0; k<update_size; k++){
        int idx2 = update_idx(k);
        eff_coeff(idx2, 2) = proposed_delta(k);
      }
    }
  }
  return eff_coeff;
}



// 5-1. delta update
MatrixXd update_delta2(MatrixXd eff_coeff,
                      const double V_delta,
                      const VectorXd& psi,
                      const double theta_omega,
                      const VectorXd& y,
                      const VectorXd& alpha0,
                      const MatrixXd& M,
                      const VectorXd& alpha,
                      const double alpha_p,
                      const VectorXd& treat,
                      const MatrixXd& X,
                      double epsilon,
                      double L){
  
  VectorXd gamma = eff_coeff.col(1);
  VectorXd delta = eff_coeff.col(2);
  VectorXd omega = eff_coeff.col(3);
  
  int q = delta.rows(), move_2;
  int g_gamma = gamma.sum(), g_omega = omega.sum();
  
  MatrixXd result1 = MatrixXd::Zero(q, 4);
  MatrixXd result2 = MatrixXd::Zero(q, 4);
  
  if(g_gamma == 0){
    
    result1 = eff_coeff;
    
  }else{
    
    if(g_omega == 0){
      move_2 = 1;
    }else if(g_omega == g_gamma){
      move_2 = 2;
    }else{
      move_2 = ran_sample(3);
    }
    
    switch(move_2){
    case 1: // add step
      result1 = add_step2(eff_coeff, V_delta, psi, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    case 2: // delete step
      result1 = delete_step2(eff_coeff, V_delta, psi, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    case 3: // swap step
      result1 = swap_step2(eff_coeff, V_delta, psi, theta_omega, y, alpha0, M, alpha, alpha_p, treat, X);
      break;
    }
  }
  
  eff_coeff = result1;
  
  // refining step
  result2 = HMC(eff_coeff, y, M, X, treat, alpha0, alpha, alpha_p, psi, epsilon, L);
  eff_coeff = result2;
  
  return eff_coeff;
}


#endif
