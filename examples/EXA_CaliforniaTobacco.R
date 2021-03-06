### EXAMPLE 1: California Tobacco Consumption
### Jeremy L Hour
### 11 Juillet 2016

########## WARNING ##########
# This code has been used for development of the main functions,
# and may not work. Results have not been checked carefully. 
# Mistakes might remain.

rm(list=ls())
set.seed(12071990)

### Load packages
library("MASS")
library("ggplot2")
library("gtable")
library("grid")
library("reshape2")
library("LowRankQP")

### Load user functions
source("functions/wsoll1.R")
source("functions/TZero.R")
source("functions/synthObj.R")

### 0. Loading data
data = data.frame(t(read.table("/Users/jeremylhour/Documents/data/MLAB_data.txt")))

Names = c("State_ID","Income","RetailPrice", "Young", "BeerCons","Smoking1988", "Smoking1980","Smoking1975",
           mapply(function(x) paste("SmokingCons",x,sep=""),1970:2000))
colnames(data) = Names
States = c("Alabama", "Arkansas","Colorado","Connecticut","Delaware",
                    'Georgia',  'Idaho',  'Illinois',  'Indiana', 'Iowa', 'Kansas',
                    'Kentucky', 'Louisiana', 'Maine', 'Minnesota', 'Mississippi',
                    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
                    'New Mexico', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
                    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
                    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia' , 'West Virginia',
                    'Wisconsin', 'Wyoming', 'California')
rownames(data) = States
data[,"Treated"]= as.numeric(data[,"State_ID"]==3) #California is state with ID=3

CASmoke = ts(unlist(data[data[,"Treated"]==1, mapply(function(x) paste("SmokingCons",x,sep=""),1970:2000)]),
                start=c(1970), freq=1)

### 1. Arbitrary tuning parameter
X = data[,c("Income","RetailPrice", "Young", "BeerCons",
                  mapply(function(x) paste("SmokingCons",x,sep=""),1970:1988))]
d = data[,"Treated"]
V = diag(ncol(X))

X0 = t(X[d==0,])
X1 = t(X[d==1,])

### Regularization path
lambda = seq(0,1,.001)
Wsol = matrix(nrow=length(lambda), ncol=sum(1-d))
att = vector(length = length(lambda))

for(i in 1:length(lambda)){
  sol = wsoll1(X0,X1,V,lambda[i])
  sol = TZero(sol)
  Wsol[i,] = sol
}

colnames(Wsol) = States[States!="California"]
print("Non-zero weights for the pure synthetic control:")
Wsol[1,Wsol[1,]!=0]

# Weight as function of penalty level
matplot(lambda,Wsol, type="l", lwd=2,
        main="Regularization path",
        xlab="Penalty level", ylab="weight", ylim=c(0,1))


# All possible counterfactuals
y = data[rownames(data)!="California",mapply(function(x) paste("SmokingCons",x,sep=""),1970:2000)]  
SyntheticControl = Wsol %*% as.matrix(y)

matplot(1970:2000,t(SyntheticControl), type="l",
        main="All possible counterfactuals",
        xlab="Year", ylab="weight", ylim=c(35,150))
lim <- par("usr")
rect(1988, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 


### 2. Cross-validation to select tuning parameter (time-based)
varname = mapply(function(x) paste("SmokingCons",x,sep=""),1970:1980)
Xtrain = cbind(data[,c("Income","RetailPrice", "Young", "BeerCons", varname)])

V = diag(ncol(Xtrain))
X0 = t(Xtrain[d==0,])
X1 = t(Xtrain[d==1,])

### Computing with all lambda's on training sample
lambda = seq(0,3,.01)
Wsol = matrix(nrow=length(lambda), ncol=sum(1-d))
att = vector(length = length(lambda))

for(i in 1:length(lambda)){
  sol = wsoll1(X0,X1,V,lambda[i])
  sol = TZero(sol)
  Wsol[i,] = sol
}

colnames(Wsol) = States[States!="California"]

### See performance on test sample
varname = mapply(function(x) paste("SmokingCons",x,sep=""),1983:1988)
Xtest = data[rownames(data)!="California",varname]
SyntheticControl = Wsol %*% as.matrix(Xtest)
MSPE = (SyntheticControl - kronecker(matrix(1,nrow=length(lambda)),as.matrix(data[rownames(data)=="California",varname])))^2
MSPE = apply(MSPE,1,mean)


matplot(lambda,MSPE, type="o", pch=20,
        main="MSPE", col="steelblue",
        xlab=expression(lambda), ylab="MSPE")

lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))])

varname = mapply(function(x) paste("SmokingCons",x,sep=""),1970:2000)
y = data[rownames(data)!="California",varname]  
SyntheticControl = t(Wsol[which(MSPE==min(MSPE)),] %*% as.matrix(y))
OriginalSC = t(Wsol[1,] %*% as.matrix(y))

plotdata = ts(cbind(t(data[rownames(data)=="California",varname]), SyntheticControl, OriginalSC),start=c(1970), freq=1)


plot(plotdata, plot.type="single",
     col=c("steelblue","firebrick","forestgreen"), lwd=2,
     lty=c(1,6,6),xlab="", ylab="Cigarette consumption (Packs per capita)",
     ylim=c(35,150))
lim <- par("usr")
rect(1988, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(1971,80,
       legend=c("Real California", "Synthetic Control, opt lambda", "Original Synthetic Control"),
       col=c("steelblue","firebrick","forestgreen"), lwd=2,
       lty=c(1,6,6))


### 3. Cross-validation (unit based)
varname = mapply(function(x) paste("SmokingCons",x,sep=""),1970:2000)
Xtrain = cbind(data[d==0,c("Income","RetailPrice", "Young", "BeerCons", varname)])
testvarname = mapply(function(x) paste("SmokingCons",x,sep=""),1989:2000)
Xtest = cbind(data[d==0,testvarname])

V = diag(ncol(Xtrain))
dstar = rep(0,sum(d==0))
lambda = seq(0,3.5,.01)
MSPE = matrix(nrow=sum(d==0), ncol=length(lambda))

for(j in 1:sum(d==0)){
  dstar[j] = 1 
  X0 = t(Xtrain[dstar==0,])
  X1 = t(Xtrain[dstar==1,])

  Wsol = matrix(nrow=length(lambda), ncol=sum(1-dstar))
  
  for(i in 1:length(lambda)){
    sol = wsoll1(X0,X1,V,lambda[i])
    sol = TZero(sol)
    Wsol[i,] = sol
  }
  
  SyntheticControl = Wsol %*% as.matrix(Xtest[dstar==0,])
  MSPE_i = (SyntheticControl - kronecker(matrix(1,nrow=length(lambda)),as.matrix(Xtest[dstar==1,])))^2
  MSPE[j,] = apply(MSPE_i,1,mean)
  
  dstar[j] = 0 
}

MSPETot = apply(MSPE,2,mean)

matplot(lambda,MSPETot, type="o", pch=20,
        main="MSPE", col="steelblue",
        xlab=expression(lambda), ylab="MSPE")


### Computing with all lambda's on training sample
varname = mapply(function(x) paste("SmokingCons",x,sep=""),1970:1980)
Xtrain = cbind(data[,c("Income","RetailPrice", "Young", "BeerCons", varname)])
X0 = t(Xtrain[d==0,])
X1 = t(Xtrain[d==1,])

lambda = seq(0,3.5,.001)
Wsol = matrix(nrow=length(lambda), ncol=sum(1-d))

for(i in 1:length(lambda)){
  sol = wsoll1(X0,X1,V,lambda[i])
  sol = TZero(sol)
  Wsol[i,] = sol
}

colnames(Wsol) = States[States!="California"]

### See performance on test sample
varname = mapply(function(x) paste("SmokingCons",x,sep=""),1983:1988)
Xtest = data[rownames(data)!="California",varname]
SyntheticControl = Wsol %*% as.matrix(Xtest)
MSPE = (SyntheticControl - kronecker(matrix(1,nrow=length(lambda)),as.matrix(data[rownames(data)=="California",varname])))^2
MSPE = apply(MSPE,1,mean)


matplot(lambda,MSPE, type="o", pch=20,
        main="MSPE", col="steelblue",
        xlab=expression(lambda), ylab="MSPE")

lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))])