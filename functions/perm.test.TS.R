#' Permutation Test for regularized Synthetic Control
#' 
#' Created: 12 avril 2017
#' Returns R ATET estimates on reshuffled samples, with outcome begin a time series
#' and compute the p-value for sharp null hypothesis with constant C.
#' Used to construct confidence intervals.
#' 
#' Especially created to deal with Acemoglu example.
#' 
#' @param d is a vector of dimension n (treatment indicator)
#' @param X is a matrix of dimension n x p
#' @param y is a vector of dimension n x k (outcome)
#' @param V is a p x p matrix of weights
#' @param lambda is a positive penalty level
#' @param R is the number of replications
#' @param C is the constant for the TE is the sharp null hypothesis
#' 
#' @author Jeremy LHour

perm.test.TS <- function(d,y,X,V,lambda,R=1000,C=0){
  # Compute ATET on original sample
  X0 = t(X[d==0,]); X1 = t(X[d==1,]);
  Y0 = y[d==0,]; Y1 = y[d==1,];
  
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
  Ypot0 = y - C*matrix(rep(d,ncol(y)),ncol=ncol(y)); Ypot1 = y + C*matrix(rep(1-d,ncol(y)),ncol=ncol(y));
  dstar = sample(d)
  X0 = t(X[dstar==0,]); X1 = t(X[dstar==1,]);
  Y0 = Ypot0[dstar==0,]; Y1 = Ypot1[dstar==1,];
  solstar = regsynth(X0,X1,as.vector(Y0[,1]),as.vector(Y1[,1]),V,lambda)
  
  sigma = sqrt(apply((X1 - X0%*%t(solstar$Wsol))^2,2,mean)) # Goodness of fit for each treated over pre-treatment period
  omega = 1/(sigma*sum(1/sigma))
  phi = cumsum((Y1 - Y0%*%t(solstar$Wsol))%*%omega)
  return(phi)
}