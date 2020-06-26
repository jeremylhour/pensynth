### Delaunay Example for Paper
### Jeremy L Hour
### 26 decembre 2018

# setwd("//ulysse/users/JL.HOUR/1A_These/A. Research/RegSynthProject/regsynth")
setwd("/Users/jeremylhour/Documents/Recherche/RegSynthProject/regsynth")
rm(list=ls())

### 0. Settings

### Load packages
library("MASS")
library("ggplot2")
library("gtable")
library("grid")
library("reshape2")
library("LowRankQP")
library("xtable")
library("deldir")
library("plotrix")

### Load user functions
source("functions/wsol.R")
source("functions/wsoll1.R")
source("functions/PolyDGP.R")
source("functions/wATT.R")
source("functions/matching.R")
source("functions/matchest.R")
source("functions/OBest.R")
source("functions/regsynth.R")
source("functions/regsynthpath.R")
source("functions/TZero.R")
source("functions/synthObj.R")
source("functions/which.tile.R")

## Drawing little arrows
arrowLine <- function(x, y, N=10, ...){
  lengths <- c(0, sqrt(diff(x)^2 + diff(y)^2))
  l <- cumsum(lengths)
  tl <- l[length(l)]
  el <- seq(0, to=tl, length=N+1)[-1]
  
  plot(x, y, t="l", ...)
  
  for(ii in el){
    
    int <- findInterval(ii, l)
    xx <- x[int:(int+1)]
    yy <- y[int:(int+1)]
    
    ## points(xx,yy, col="grey", cex=0.5)
    
    dx <- diff(xx)
    dy <- diff(yy)
    new.length <- ii - l[int]
    segment.length <- lengths[int+1]
    
    ratio <- new.length / segment.length
    
    xend <- x[int] + ratio * dx
    yend <- y[int] + ratio * dy
    points(xend,yend, col="white", pch=19)
    arrows(x[int], y[int], xend, yend, length=0.1)
    
  }
}

### MC XP
set.seed(2121988)
lambda = seq(.001,30,.1) # set of lambda to be considered for optim
n1 = 1
n0 = 100
p = 2
delta = 2

# Setting up data
data = PolyDGP(n1,n0,p,delta)
X = data$X; y = data$y; d = data$d

X0 = t(X[d==0,]); X1 = t(X[d==1,]); V = diag(ncol(X))
Y0 = y[d==0]; Y1 = y[d==1]; n0 = sum(1-d)


# Find Delaunay tesselation (control only and control+treatd)
DTco = deldir(X0[1,], X0[2,])
DTct = deldir(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]))


############################
############################
############################
### BEGINNING OF EXAMPLE ###
############################
############################
############################

# Take 100 control units distributed uniformly in [0,1] x [0,1]
# An 1 treated unit with same distribution.
X1 = matrix(c(.15,.385),ncol=2)

### Panel A: Dots and Delaunay triangulation
pdf(file="plot/DTPanelA.pdf", height=10, width=10)
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=1, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
dev.off()

# Let's represent the synthetic control units for a wide range of lambda's.
# Synthetic control for each lambda
solpath = regsynthpath(X0,t(X1),Y0,Y1,diag(p),lambda)

W0 = drop(solpath$Wsol)
SyntheticUnit = W0%*%t(X0)

# Find the nearest neighbor
NN = matching(X0,t(X1),diag(p),m=1)
Xnn = X0%*%NN

# Represent NN circle
rad = sqrt(sum((t(X1)-Xnn)^2))

### Panel B: with synthetic solutions as lambda moves
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=1, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
draw.circle(X1[,1], X1[,2],radius=rad,nv=100,border="black",lty=1,lwd=2)

# What are the active control units?
# used in synthetic controls (across all lambdas)
active = which(apply(W0,2,sum)>0)

### Panel C: what units are active?
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=1, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
points(t(X0[,active]), pch=16, col="black", cex=1, lwd=5)
points(t(Xnn), pch=16, col="black", cex=1, lwd=5)

# If we create the Augmented Delaunay Triangulation (adding the treated unit),
# we can determine who might be active or not.
# Find Delaunay tesselation (control only and control+treatd)
DTct = deldir(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]))

# Connections in augmented DT
# treated unit is index n0+1
union(DTct$delsgs[DTct$delsgs[,"ind2"]==n0+1,"ind1"],
      DTct$delsgs[DTct$delsgs[,"ind1"]==n0+1,"ind2"])

# Panel D
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=1, lwd=5)
plot(DTct, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
points(t(X0[,active]), pch=16, col="black", cex=1, lwd=5)
points(t(Xnn), pch=16, col="black", cex=1, lwd=5)


### Putting everything together
### Panel A: Dots and Delaunay triangulation
pdf(file="plot/DTPanelA.pdf", height=6, width=6)
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
#points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=2, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
dev.off()

### Panel B: with synthetic solutions as lambda moves
### Draws arrows

pdf(file="plot/DTPanelB.pdf", height=6, width=6)
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
#points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=2, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
draw.circle(X1[,1], X1[,2],radius=rad,nv=100,border="black",lty=1,lwd=2)
dev.off()

### Panel C: what units are active?
pdf(file="plot/DTPanelC.pdf", height=6, width=6)
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
#points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=2, lwd=5)
plot(DTco, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
points(t(X0[,active]), pch=16, col="black", cex=2, lwd=1)
points(t(Xnn), pch=16, col="black", cex=2, lwd=1)
dev.off()

# Panel D
pdf(file="plot/DTPanelD.pdf", height=6, width=6)
plot(c(X0[1,],X1[,1]), c(X0[2,],X1[,2]), type="n", asp=1,xlim=c(0.2,.3),ylim=c(.2,.6),
     xlab="",ylab="",xaxt='n', ann=FALSE, yaxt='n')
#points(X0[1,], X0[2,], pch=1, col="black", cex=1, lwd=2)
points(X1[,1], X1[,2], pch=4, col="black", cex=2, lwd=5)
plot(DTct, wlines="triang", wpoints="none", number=FALSE, add=TRUE, lty=2)
points(SyntheticUnit,type="line",col="black", lwd=6, lty=1)
points(t(X0[,active]), pch=16, col="black", cex=2, lwd=1)
points(t(Xnn), pch=16, col="black", cex=2, lwd=1)
dev.off()