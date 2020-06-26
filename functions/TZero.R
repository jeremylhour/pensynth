#' TZero
#' 
#' Auxiliary function to convert small values in zeros
#' 
#' @param x vector with positive elements
#' @param tol tolerance level
#' @param scale If TRUE rescale so new vector elements sum to one

TZero <- function(x,tol=1e-6,scale=T){
  if(!all(x > 0)) stop("Some elements are negative!")
  y = ifelse(x < tol,0,x)
  if(scale) y = y/sum(y)
  return(y)
}