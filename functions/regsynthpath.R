#' Regularized Synthetic Control path
#' 
#' Created: 20 juillet 2016
#' Modified: 13 mars 2018
#' Loops over regsynth to get regularized synthetic control solutions
#' For all given values of lambda
#' 
#' @param X0 a p x n0 matrix
#' @param X1 a p x n1 vector
#' @param Y0 a n0 x 1 vector
#' @param Y1 a n1 x 1 vector
#' @param V a p x p matrix of weights
#' @param lambda a vector of l1 penalty levels
#' @param tol gives the threshold for considering true zeros
#' @param bar if TRUE displays progress bar
#' 
#' @seealso \code{\link{regsynth}}
#' 
#' @author Jeremy L'Hour

regsynthpath <- function(X0,X1,Y0,Y1,V,lambda,tol=1e-6,bar=F){
  K = length(lambda); n0 = ncol(X0); n1 = ncol(X1)
  ATT = vector(length = K)
  tau = matrix(nrow=K, ncol=n1)
  Wsol = array(dim=c(K,n1,n0))
  
  t_start = Sys.time()
  if(bar) pb = txtProgressBar(style = 3)
  for(k in 1:K){
    sol = regsynth(X0,X1,Y0,Y1,V,lambda[k],tol=1e-6)
    ATT[k] = sol$ATT
    tau[k,] = sol$CATT
    Wsol[k,,] = sol$Wsol
    if(bar) setTxtProgressBar(pb, k/K)
  }
  if(bar) close(pb)
  print(Sys.time()-t_start)
  
  return(list(ATT=ATT,
              CATT=tau,
              Wsol=Wsol))
}
