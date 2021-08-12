#' Confidence intervals
#' Penalized Synthetic control
#' 21 octobre 2016
# @author : jeremylhour

rm(list=ls())

### 0. Settings

### Load packages
library("MASS")
library("ggplot2")
library("gtable")
library("grid")
library("reshape2")
library("LowRankQP")

### Load user functions
source("functions/matchDGP.R")
source("functions/regsynth.R")
source("functions/conf.interval.R")

### 0. Generate data
set.seed(12071990)
data = matchDGP(n=100,p=10,Ry=.5,Rd=.2,a=0)
X = data$X; y = data$y; d = data$d

X0 = t(X[d==0,]); X1 = t(X[d==1,]); V = diag(ncol(X))
Y0 = y[d==0]; Y1 = y[d==1]; n0 = sum(1-d)

P = 2*t(X0)%*%V%*%X0
sum(eigen(P)$values < 0)

sol1 = regsynth(X0,X1,y[d==0],y[d==1],V,.1)
theta.obs = sol1$ATT
print(theta.obs)

# Get confidence interval
conf.interval(d,y,X,diag(ncol(X)),lambda=.1,B=25000,alpha=.05)