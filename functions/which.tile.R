#' which.tile
#' 
#' Finds the Dirchlet/Voronoi tile of a tessellation produced by deldir that contains a given point.
#' 
#' @param x The x coordinate of the point in question.
#' @param y The y coordinate of the point in question.
#' @param tl	A tile list, as produced by the function tile.list() from a tessellation produced by deldir().

which.tile <- function(x,y,tl){
  u  <- c(x,y)
  nt <- length(tl)
  d2 <- numeric(nt)
  for(i in 1:nt) {
    d2[i] <- sum((u-tl[[i]]$pt)^2)
  }
  which.min(d2)
}