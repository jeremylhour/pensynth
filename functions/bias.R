#' Bias
#' 
#' Created: 08 aout 2018
#' Compute Linear Bias for Bias Correction
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x n1 vector
#' @param y is a T x (n1+n0) matrix
#' @param d is a n1+n0 vector
#' @param W is a n1 x n0 vector of weights
#' 
#' @seealso \code{\link{regsynth}}
#' 
#' @author Jeremy L'Hour

bias <- function(X0,X1,y,d,W){
  n1 = ncol(X1); n0 = ncol(X0)
  reg0 = rbind(X0,rep(1,n0))
  reg1 = rbind(X1,rep(1,n1))
  A = solve(reg0%*%t(reg0))%*%reg0
  
  bias = matrix(nrow=nrow(y),ncol=n1)
  for(t in 1:nrow(y)){
    mu0 = A%*%y[t,d==0]
    bias[t,] = t(reg1)%*%mu0 - W%*%(t(reg0)%*%mu0)
  }
  return(bias)
}
  