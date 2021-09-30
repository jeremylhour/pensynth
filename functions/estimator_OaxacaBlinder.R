#' Oaxaca-Blinder estimator for a whole sample
#' 
#' From a sample of treated and control units, computes the weights
#' for each counterfactual, the individual treatment effects and counterfactuals,
#' and the Average Treatment on the Treated (ATT). Based on Kline (2011).
#' 
#' Edited: 13 octobre 2016
#' 
#' @param d is a vector of dimension n (treatment indicator)
#' @param X is a n x p matrix
#' @param y is a vector of dimension n (outcome)
#' 
#' @return ATT is the Average Treatment Effect on the Treated over the sample.
#' @return CATT is the individual treatment effect.
#' @return Wsol is the n1 x n0 matrix of weights to compute counterfactual.
#' @return y0_hat is the individual counterfactual for each treated unit.
#' 
#' @seealso \code{\link{matching}} for the computation of the matching weights.
#' 
#' @author Jeremy L'Hour

OBest <- function(d,X,y){
  
  X0 = t(X[d==0,]); X1 = t(X[d==1,]);
  Y0 = y[d==0]; Y1 = y[d==1]; 
  n1 = sum(d); n0 = sum(1-d);
  
  Wsol = t(X1) %*% solve(X0 %*% t(X0)) %*% X0
  y0_hat = Wsol%*%Y0
  tau = Y1 - y0_hat
  
  return(list(ATT = mean(tau),
              CATT = tau,
              Wsol = Wsol,
              y0_hat = y0_hat))
}