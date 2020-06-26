#' Permutation Test for regularized Synthetic Control
#' 
#' Created: 9 septembre 2016
#' Returns R ATET estimates on reshuffled samples
#' and compute the p-value for sharp null hypothesis with constant C.
#' Used to construct confidence intervals.
#' 
#' @param d is a vector of dimension n (treatment indicator)
#' @param X is a matrix of dimension n x p
#' @param y is a vector of dimension n (outcome)
#' @param V is a p x p matrix of weights
#' @param lambda is a positive penalty level
#' @param R is the number of replications
#' @param C is the constant for the TE is the sharp null hypothesis
#' 
#' @author Jeremy LHour

perm.test <- function(d,y,X,V,lambda,R=1000,C=0){
  # Compute ATET on original sample
  X0 = t(X[d==0,]); X1 = t(X[d==1,]);
  Y0 = y[d==0]; Y1 = y[d==1];
  sol1 = regsynth(X0,X1,y[d==0],y[d==1],V,lambda)
  theta.obs = sol1$ATT
  
  # Compute ATET on as many reshuffled samples
  theta.reshuffled = replicate(R, permutation.iter.C(d,y,X,V,lambda,C), simplify="vector")
  
  # Compute p-value
  p.val = mean(abs(theta.reshuffled - C) >= abs(theta.obs-C))
  print(paste("P-value: ",p.val))
  
  return(list(p.val=p.val,
              theta.obs=theta.obs,
              theta.reshuffled=theta.reshuffled))
}


### Auxiliary function
permutation.iter.C = function(d,y,X,V,lambda,C=0){
  Ypot0 = y - d*C; Ypot1 = y + (1-d)*C;
  dstar = sample(d)
  X0 = t(X[dstar==0,]); X1 = t(X[dstar==1,]);
  Y0 = Ypot0[dstar==0]; Y1 = Ypot1[dstar==1];
  solstar = regsynth(X0,X1,Y0,Y1,V,lambda)
  return(solstar$ATT)
}