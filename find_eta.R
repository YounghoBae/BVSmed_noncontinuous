##---------------------------------------------------------------
## Required libraries
##---------------------------------------------------------------

#----- Parallel computing
library(doParallel)

#----- Make clusters based on the number of CPU cores
cl<-makeCluster(40)
registerDoParallel(cl)
getDoParWorkers()

#----- Support parallel excution
library(foreach)

##---------------------------------------------------------------
## 'Main' function on each cluster (parallel)
##---------------------------------------------------------------

main <- function(temp){
  
  library(MASS)
  library(invgamma)
  library(dplyr)
  library(mvtnorm)
  library(Rcpp)
  sourceCpp("code_main/main.cpp")
  load("data/data.RData")
  
  outcome = cmaVSm(treat=treat, M=M, X=X, y=y, iter=100000, burn_in=70000, thin=30,
                   theta_gamma=-2.2, mu_lambda=0, h_lambda=100, nu_element=3,
                   h0=100, c0=100, V_tau=0.01, V_lambda=0.01, mu0=0, mu1=0, nu0=6, sigmaSq0=1/3, 
                   update_prop=c(1,0,1,1,0,0,0,1,1,0), eta = temp,
                   init1 = 0.1, init2 = 0.1, init3 = 0.1, init4 = 0.1, init5 = 0.1)
  
  return(outcome$post_gamma)
}

temp_sub = seq(0, 0.975, 0.025)
data <- foreach(temp = temp_sub) %dopar% main(temp)

save(data, file="data.RData")
stopCluster(cl)

result = matrix(NA, 40, 1)
rownames(result) = seq(0, 0.975, 0.025)
colnames(result) = "post_gamma"
for(i in 1:40){
  result[i,1] = quantile(data1[[i]], 0.5)
}

# rownames are eta, post_gamma is corresponding 50th post_gamma
result

