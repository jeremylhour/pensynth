#' Synthetic Control
#' 
#' Function to compute synthetic control weights given a V matrix
#' Also implements L2 penalty
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x 1 vector
#' @param V is a p x p matrix of weights
#' @param pen is l2 penalty level
#' 
#' @autor Jeremy LHour

wsoll2 <- function(X0,X1,V,pen=0.0){
  n = ncol(X0)
  dis = X0 - matrix(1, ncol=n) %x% X1
  
  P = 2*t(X0)%*%V%*%X0 + diag(diag(2*pen*t(dis)%*%V%*%dis))
  q = t(-2*t(X0)%*%V%*%X1)
  
  sol = LowRankQP(Vmat=P,dvec=q,Amat=matrix(1, ncol=n),bvec=1,uvec=rep(1,n), method="LU")
  return(sol$alpha)
}