#' Penalized synthetic control weights
#' 
#' Function to compute synthetic control weights given a V matrix
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x 1 vector
#' @param V is a p x p matrix of weights
#' @param pen L1 penalty level (lambda)
#' 
#' @author Jérémy L'Hour

wsoll1 <- function(X0,X1,V,pen=0.0){
  n = ncol(X0)
  Delta = diag(t(X0 - matrix(rep(1,n),ncol=n)%x%X1)%*%V%*%(X0 - matrix(rep(1,n),ncol=n)%x%X1))
  
  P = 2*t(X0)%*%V%*%X0
  q = t(-2*t(X0)%*%V%*%X1 + pen*Delta)
  
  sol = LowRankQP(Vmat=P,dvec=q,Amat=matrix(1, ncol=n),bvec=1,uvec=rep(1,n), method="LU")
  return(sol$alpha)
}