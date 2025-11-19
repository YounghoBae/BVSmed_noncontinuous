#ifndef __ALPHA_H__
#define __ALPHA_H__

#include <Rcpp.h>
#include <RcppEigen.h>
#include "basic.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//[[Rcpp::depends(RcppEigen)]]

// Library Functions
VectorXd update_alpha0(const VectorXd& y, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, const VectorXd& alpha_0, const VectorXd& alpha, double alpha_p, const VectorXd& treat, double s0, double V_s);

VectorXd update_alpha(const VectorXd& y, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, const VectorXd& alpha_0, const VectorXd& alpha, double alpha_p, const VectorXd& treat, double t0, double V_t);

double update_alpha_p(const VectorXd& y, const MatrixXd& M, const VectorXd& delta, const MatrixXd& X, const VectorXd& alpha_0, const VectorXd& alpha, double alpha_p, const VectorXd& treat, double k0, double V_k);

// Functions Define
// 1. alpha0 update
VectorXd update_alpha0(const VectorXd& y,
                       const MatrixXd& M,
                       const VectorXd& delta,
                       const MatrixXd& X,
                       const VectorXd& alpha_0,
                       const VectorXd& alpha,
                       double alpha_p,
                       const VectorXd& treat,
                       double s0,
                       double V_s) {
  
  int n = alpha_0.rows();
  VectorXd out1 = VectorXd::Zero(n);
  double new_alpha_0_val = Rcpp::rnorm(1, alpha_0(0), sqrt(V_s))(0);
  VectorXd new_alpha_0 = VectorXd::Constant(n, new_alpha_0_val);
  
  double log_prior1 = - (0.5 / s0) * pow(alpha_0(0), 2);
  double log_prior2 = - (0.5 / s0) * pow(new_alpha_0_val, 2);
  
  double log_like1 = loglike_binary(y, delta, M, X, treat, alpha_0, alpha, alpha_p);
  double log_like2 = loglike_binary(y, delta, M, X, treat, new_alpha_0, alpha, alpha_p);
  
  double update_prob = log_like2 + log_prior2 - log_like1 - log_prior1;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(update_prob > log_u){
    out1 = new_alpha_0;
  }else{
    out1 = alpha_0;
  }
  
  return out1;
}

// 2. alpha update
VectorXd update_alpha(const VectorXd& y,
                      const MatrixXd& M,
                      const VectorXd& delta,
                      const MatrixXd& X,
                      const VectorXd& alpha_0,
                      const VectorXd& alpha,
                      double alpha_p,
                      const VectorXd& treat,
                      double t0,
                      double V_t) {

  int p = alpha.rows();
  VectorXd new_alpha = alpha;
  
  for(int i=0; i<p; i++){
    new_alpha(i) = Rcpp::rnorm(1, alpha(i), sqrt(V_t))(0);
    
    double log_prior1 = - (0.5 / t0) * pow(alpha(i), 2);
    double log_prior2 = - (0.5 / t0) * pow(new_alpha(i), 2);
    
    double log_like1 = loglike_binary(y, delta, M, X, treat, alpha_0, alpha, alpha_p);
    double log_like2 = loglike_binary(y, delta, M, X, treat, alpha_0, new_alpha, alpha_p);
    
    double update_prob = log_like2 + log_prior2 - log_like1 - log_prior1;
    double log_u = log(Rcpp::runif(1)(0));
    
    if(update_prob < log_u){
      new_alpha(i) = alpha(i);
    }
  }
  
  return new_alpha;
}

// 3. alpha prime update
double update_alpha_p(const VectorXd& y,
                      const MatrixXd& M,
                      const VectorXd& delta,
                      const MatrixXd& X,
                      const VectorXd& alpha_0,
                      const VectorXd& alpha,
                      double alpha_p,
                      const VectorXd& treat,
                      double k0,
                      double V_k) {
  
  double out1;
  double new_alpha_p = Rcpp::rnorm(1, alpha_p, sqrt(V_k))(0);
  
  double log_prior1 = - (0.5 / k0) * pow(alpha_p, 2);
  double log_prior2 = - (0.5 / k0) * pow(new_alpha_p, 2);
  
  double log_like1 = loglike_binary(y, delta, M, X, treat, alpha_0, alpha, alpha_p);
  double log_like2 = loglike_binary(y, delta, M, X, treat, alpha_0, alpha, new_alpha_p);
  
  double update_prob = log_like2 + log_prior2 - log_like1 - log_prior1;
  double log_u = log(Rcpp::runif(1)(0));
  
  if(update_prob > log_u){
    out1 = new_alpha_p;
  }else{
    out1 = alpha_p;
  }
  
  return out1;
}

#endif