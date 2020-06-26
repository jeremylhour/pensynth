#' Penalized synthetic control function
#' 
#' Main function: compute synthetic control for a given value of lambda
#' Allows parallel
#' 
#' Created: 20 juillet 2016
#' Edited: 2 avril 2020
#' Returns ATT, Conditional ATT, counterfactual and n1 sets of n0 weights from regularized Synthetic Control
#' 
#' @param X0 a p x n0 matrix
#' @param X1 a p x n1 matrix
#' @param Y0 a n0 x 1 vector
#' @param Y1 a n1 x 1 vector
#' @param V a p x p matrix of weights (optionnal)
#' @param pen L1 penalty level (lambda)
#' @param tol gives the threshold for considering true zeros
#' @param if TRUE use parallel version careful not to use it in a loop
#' 
#' @seealso \code{\link{wsoll1}}
#' 
#' @author Jeremy L'Hour

regsynth <- function(X0,X1,Y0,Y1,V,pen,tol=1e-6,parallel=FALSE){
  if(missing(V)) V = diag(nrow(X0))
  n0 = ncol(X0); n1 = ncol(X1)
  func_list = c("wsoll1","TZero")
  
  if(parallel){
    # start parallel
    cores = detectCores()
    cl = makeCluster(5, outfile="",setup_timeout=.5)
    registerDoParallel(cl)
    t_start <- Sys.time()
    res <- foreach(i = 1:n1,.export=func_list,.packages=c('LowRankQP'),.combine='cbind', .multicombine=TRUE, .errorhandling = 'remove') %dopar% {
      sol = wsoll1(X0,X1[,i],V,pen)
      sol = TZero(sol,tol)
      sol
    }
    stopCluster(cl)
    print('--- Computation over ! ---')
    print(Sys.time()-t_start)
    Wsol = t(res)
  } else {
    Wsol = matrix(nrow=n1,ncol=n0)
    f = file()
    sink(file=f)
    for(i in 1:n1){
      sol = wsoll1(X0,X1[,i],V,pen)
      sol = TZero(sol,tol)
      Wsol[i,] = sol
    }
    sink()
    close(f)
  }

  y0_hat = Wsol%*%Y0
  tau = Y1 - y0_hat
  return(list(ATT = mean(tau),
              CATT = tau,
              Wsol = Wsol,
              y0_hat = y0_hat))
}
