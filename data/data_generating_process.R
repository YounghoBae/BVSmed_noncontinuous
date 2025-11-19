## Load library
library(MASS)

## Load required data
load("data/covM_reorder.RData")

## Setting random seed
set.seed(16)

## Data generating process
# n : # of observations, q : # of mediators, p : # of confounders
n = 466; q = 298; p = 3

# X : confounders matrix
X = mvrnorm(n, mu = rep(0,p), Sigma = diag(p))

# treat : vector of treatment variable
treat = rnorm(n, 0.5*X[,1]+0.2*X[,2]+0.7*X[,3], 1)

beta0_vec = rep(0.1,q)
B         = matrix(0.1, nrow = p, ncol = q)

# setting true tau vector that is selected in high correaltion mediators group based on covariance matrix
tau = c(rep(c(-0.12, -0.08, -0.04, 0.04, 0.08, 0.12), each = 5), rep(0, q-30))

M = matrix(0,n,q)
for(i in 1:n){
  M[i,] = mvrnorm(1, beta0_vec + tau*treat[i] + (t(B) %*% X[i,]), 0.5*covM_reorder)
}

# setting true delta vector that is selected in high correaltion mediators group based on covariance matrix
delta = c(rep(c(2, 3, 4, 0, 0), 6), rep(0, q-30))

alpha0   = -2.5
alpha    = rep(2, p)
alpha_p  = 2

y = rep(0,n)
A = alpha0 + M %*% delta + X %*% alpha + alpha_p * treat
pi_A =  1 / (1 + exp(-A))

y = sapply(pi_A, function(p) sample(c(0,1),1,prob=c(1-p,p)))

save(treat, M, X, y, file = "data.RData")
