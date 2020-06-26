#' Pure Synthetic Control
#' 
#' Function to compute pure synthetic control weights using Delaunay Triangulation
#' Warning: Can take a long time in large dimension.
#' 
#' @param X0 is a p x n0 matrix
#' @param X1 is a p x n1 vector
#' 
#' @author Jeremy L'Hour

pure.wsol <- function(X0,X1){
  DT = delaunayn(t(X0)) # Compute Delaunay Triangulation of untreated units
  Pos = tsearchn(t(X0), DT, t(X1))
  
  n1 = ncol(X1); n0 = ncol(X0)
  sol = matrix(0, nrow=n1,ncol=n0)
  for(i in 1:n1){
    if(is.na(Pos$idx[i])){
      
    } else {
      index = DT[Pos$idx[i],]
      sol[i,index] = Pos$p[i]
    }
  }

  return(sol)
}