#' Penalized Synthetic Control, for multiple lambda
#' parallelized version
#' 
#' Created: 31 mars 2020
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

pensynth_parallel <- function(X0,X1,Y0,Y1,V,lambda,tol=1e-6){
  if(missing(V)) V = diag(nrow(X0))
  
  func_list = c("regsynth","wsoll1","TZero")
  K = length(lambda); n0 = ncol(X0); n1 = ncol(X1)
  
  # start parallel
  cores = detectCores()
  cl = makeCluster(cores[1]/2)
  registerDoParallel(cl)
  t_start <- Sys.time()
  res <- foreach(k = 1:K,.export=func_list,.packages=c('LowRankQP'),.combine='rbind', .multicombine=TRUE, .errorhandling = 'remove') %dopar% {
    sol = regsynth(X0,X1,Y0,Y1,V,lambda[k],tol=1e-6,parallel=FALSE)
    sol$Wsol
  }
  print('--- Computation over ! ---')
  print(Sys.time()-t_start)
  stopCluster(cl)
  
  # processing results
  Wsol = array(dim=c(K,n1,n0))
  tau = matrix(nrow=K, ncol=n1)
  ATT = vector(length = K)
  
  for(k in 1:K){
    Wsol[k,,] = res[((k-1)*n1+1):(k*n1),]
    tau[k,] = Y1 - Wsol[k,,]%*%Y0 
    ATT[k] = mean(tau[k,])
  }
  
  return(list(ATT=ATT,
              CATT=tau,
              Wsol=Wsol))
}

