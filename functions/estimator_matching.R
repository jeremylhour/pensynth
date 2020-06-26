#' Matching estimator for a whole sample
#' 
#' From a sample of treated and control units, computes the weights
#' for each counterfactual, the individual treatment effects and counterfactuals,
#' and the Average Treatment on the Treated (ATT). Based on Neirest-Neighbor matching.
#' 
#' Edited: 28 octobre 2016
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x n1 matrix
#' @param Y0 is a n0 x 1 vector
#' @param Y1 is a n1 x 1 vector
#' @param V is a p x p matrix of weights
#' @param m number of neighbors to find
#' 
#' @return ATT is the Average Treatment Effect on the Treated over the sample.
#' @return CATT is the individual treatment effect.
#' @return Wsol is the n1 x n0 matrix of weights to compute counterfactual.
#' @return y0_hat is the individual counterfactual for each treated unit.
#' 
#' @seealso \code{\link{matching}} for the computation of the matching weights.
#' 
#' @author Jeremy L'Hour

matchest <- function(X0,X1,Y0,Y1,V,m=3){
  n1 = ncol(X1); n0 = ncol(X0);
  
  if(missing(Y0)) Y0 = rep(0,n0)
  if(missing(Y1)) Y1 = rep(0,n1)
  if(missing(V))  V = diag(nrow(X0))
  
  Wsol = matrix(nrow=n1,ncol=n0)
  
  for(i in 1:n1){
    sol = matching(X0,X1[,i],V,m=m)
    Wsol[i,] = sol
  }
  
  y0_hat = Wsol%*%Y0
  tau = Y1 - y0_hat
  
  return(list(ATT = mean(tau),
              CATT = tau,
              Wsol = Wsol,
              y0_hat = y0_hat))
}

#' Matching function
#' 
#' Returns a vector of weights for each control unit:
#' 1/m for the m closest to X1, and 0 otherwise.
#' 
#' Edited: 12 octobre 2016
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x 1 vector
#' @param V is a p x p matrix of weights
#' @param m number of neighbors to find
#' 
#' @author Jeremy L'Hour

matching <- function(X0,X1,V,m=3){
  n = ncol(X0)
  Delta = matrix(t(X1)%*%V%*%X1, nrow=n, ncol=1) - 2*t(X0)%*%V%*%X1 + diag(t(X0)%*%V%*%X0) # Compute the distance
  r = rank(Delta,ties.method = "max")
  # handle ex-aequo: e.g. take the first two if the first are ex-aqueo
  m_valid = min(r[r>=m])
  # I take the first m_valid
  sol = ifelse(r <= m_valid,1/m_valid,0)
  return(sol)
}