#' DGP for Exp 5 
#' See e-mail from 9/3/2018 by Alberto
#' 
#' CREATED: 10/3/2018
#' 
#' @param n1 number of treated
#' @param n0 number of non treated
#' @param p is the number of covariates
#' @param delta is the order of the polynomial (should be an integer)
#' @param a is the inf bound for treated support
#' @param b is the sup bound for treated support
#' 
#' @author Jeremy LHour

PolyDGP <- function(n1=25,n0=50,p=3,delta=2,a=.1,b=.9){
  d = c(rep(1,n1),rep(0,n0))
  X = rbind(matrix(runif(n1*p,min=a,max=b), ncol=p, nrow=n1),
            matrix(runif(n0*p), ncol=p, nrow=n0))
  # Treated ~ uniform density [a;b]
  # Controls ~ uniform density [0;1]
  # all iid (across i and j)
  
  ### Create outcome function
  stdev = CNorm(p,delta,a,b)
  beta = rep(1,p)/stdev
  
  y = (X^delta)%*%beta + rnorm(n1+n0)
  
  return(list(X=X,y=y,d=d))
}

# Function to compute the normalization constant
# Make Variance of the part in X equal to one
CNorm = function(p,delta,a=0,b=1){
  CNorm = (b^(2*delta+1)-a^(2*delta+1))/((b-a)*(2*delta+1)) - (b^(delta+1)-a^(delta+1))^2/((b-a)*(delta+1))^2
  return(sqrt(p*CNorm))
}