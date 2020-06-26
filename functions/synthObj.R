#' Synthetic Control Objective function
#' 
#' Return loss and penalty part from a given set of weights
#' 
#' @param w is a n0 vector of weights
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x 1 vector
#' @param V is a p x p matrix of weights
#' @param pen is l2 penalty level
#' 
#' @autor Jeremy LHour


synthObj <- function(w,X0,X1,V){
  n = ncol(X0)
  Delta = matrix(t(X1)%*%V%*%X1, nrow=n, ncol=1) - 2*t(X0)%*%V%*%X1 + diag(t(X0)%*%V%*%X0)
  
  #1. l1-norm
  l1norm = t(Delta)%*%w
  
  #2. loss function
  loss = t(X1 - X0%*%w) %*% V %*% (X1 - X0%*%w)

  return(list(loss=c(loss),
              l1norm=c(l1norm),
              Delta=Delta))
}