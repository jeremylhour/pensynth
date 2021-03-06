### EXAMPLE 4: Geithner connections
### Jeremy L Hour
### 11 avril 2017
### EDITED: 8/8/2018

########## WARNING ##########
# This code has been used for development of the main functions,
# and may not work. Results have not been checked carefully. 
# Mistakes might remain.

setwd("//ulysse/users/JL.HOUR/1A_These/A. Research/RegSynthProject/regsynth")

rm(list=ls())

### Load packages
library("MASS")
library("ggplot2")
library("gtable")
library("grid")
library("reshape2")
library("LowRankQP")
library("R.matlab")
library("stargazer")

### Load user functions
source("functions/wsoll1.R")
source("functions/regsynth.R")
source("functions/regsynthpath.R")
source("functions/TZero.R")
source("functions/synthObj.R")
source("functions/perm.test.R")
source("functions/conf.interval.R")
source("functions/conf.interval.Geithner.R")
source("functions/bias.R")

### 0. Loading data
data = readMat("//ulysse/users/JL.HOUR/1A_These/A. Research/RegSynthProject/regsynth/data/GeithnerConnexions/Matlab Files/Data.mat")


### 1. Data Cleaning and Setting
ticker = data$ticker # firms tickers

# collect names and tickers
FirmID = data.frame()
for(i in 1:603){
  if(length(unlist(ticker[i])) > 0 & length(unlist(ticker[603+i])) > 0){
    FirmID[i,"Ticker"] = unlist(ticker[i])
    FirmID[i,"Name"] = unlist(ticker[603+i])
  } else {
    FirmID[i,"Ticker"] = "Unknown"
    FirmID[i,"Name"] = "Unknown"
  }
}

X = data.frame(data$num) # firms characteristics
names(X) = c(unlist(data$VarNames)) # setting variable names
row.names(X) = FirmID[,"Ticker"] # setting firms name

ind = is.na(X[,8]) | is.na(X[,9]) | is.na(X[,10])  # eliminating firms with no data for 'ta2008_log','roe2008','tdtc2008'
X = X[!ind,]
y = as.matrix(data$Re) # Returns
y = y[,!ind]

y[is.na(y)] = 0 # replacing missing returns with zero

# Identification of the event
ConnMeasure = 3 # 1: Shared Board 2: NY Connection 3: Geithner Schedule 4: Geithner Schedule 2007, position in data frame
GeiNomDate = 355 # Geithner nomination date
EventDate = GeiNomDate-1 
PreTreatPeriod = (GeiNomDate-281):(GeiNomDate-32) # Window of 250 days ending 30 days prior to Geithner nomination
FalsifTest = c(340:353,357:389,395:447) # Window for falsification test

# Correlation with Citi and BoA on Pre-treatment period
# We want to exclude the effect of the CitiGroup bailout
# No need to exclude BoA (tax problem after nomination, check page 29)
Citi = which(X[,5]==140)  # Citi Group 
corrCiti = cor(y[PreTreatPeriod,Citi], y[PreTreatPeriod,])
# Compute Q10 for correlation distributions
corrCitiTr = sort(corrCiti,decreasing=T)[58]

X = X[corrCiti<corrCitiTr,]
y = y[,corrCiti<corrCitiTr]

# Treatment variable
d = X[,ConnMeasure] > 0 # one or more meetings in 2007-08

# Who are the treated?
print("Treated firms and number of meetings in 2007-08")
cbind(FirmID[match(rownames(X[d==1,]), FirmID[,1]),2], X[d==1,ConnMeasure])

# Control variable other than pre-treatment outcomes, useless for now
# Include:
# - ta2008_log : firm size
# - roe2008 : profitability
# - tdtc2009 : leverage
Z = cbind(X[,c(8,9,10)], X[,c(8,9,10)]^2, X[,c(8,9,10)]^3)


### 2. Some Descriptive Statistics (TO BE CONTINUED ?)
ConnReturns = ts(apply(y[PreTreatPeriod,d==1],1,mean),start=c(1), freq=365)
NConnReturns = ts(apply(y[PreTreatPeriod,d==0],1,mean),start=c(1), freq=365)

# Balance check
# Treated
apply(X[d==1,c(8,9,10)],2,summary)
# Control
apply(X[d==0,c(8,9,10)],2,summary)

### 3. CV for selecting optimal lambda
X0 = y[PreTreatPeriod,d==0]; X1 = y[PreTreatPeriod,d==1]
Y0 = y[GeiNomDate+1,d==0]; Y1 = y[GeiNomDate+1,d==1]

V = diag(1/diag(var(t(y[PreTreatPeriod,])))) # Reweight by inverse of variance


lambda = c(seq(0,0.1,.0025),seq(0.1,2,.1)) # sequence of lambdas to test
estval = regsynthpath(X0,X1,Y0,Y1,V,lambda,tol=1e-6)
MSPE = vector(length=length(lambda))

for(k in 1:length(lambda)){
  MSPE[k] = mean(apply((y[324:354,d==1] - y[324:354,d==0]%*%t(estval$Wsol[k,,]))^2,2,mean))
}
lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))]) # Optimal lambda is .1-.2

### Figure 1: MSPE
pdf("plot/Geithner_MSPE.pdf", width=6, height=6)
matplot(lambda, MSPE, type="b", pch=20, lwd=1,
        main=expression("MSPE, "*lambda^{opt}*"= .1, computed on 30-day window"), col="steelblue",
        xlab=expression(lambda), ylab="MSPE")
abline(v=lambda.opt.MSPE,lty=2,lwd=2,col="grey")
dev.off()


### 4. Estimation

## 4.1 Penalized Synthetic Control
Psol = estval$Wsol[which(MSPE==min(MSPE)),,]
colnames(Psol) = rownames(X[d==0,])

# Number of active controls
apply(Psol>0,1,sum)
print("mean nb. active control units"); mean(apply(Psol>0,1,sum))

# Compute the statistics (see paper)
TestPeriod = (GeiNomDate-15):(GeiNomDate+30)
phiP = apply((y[TestPeriod,d==1] - y[TestPeriod,d==0]%*%t(Psol)),1,mean)
phiP_bc = phiP - apply(bias(X0,X1,y[TestPeriod,],d,Psol),1,mean) # bias corrected

sigma = sqrt(apply((X1 - X0%*%t(Psol))^2,2,mean)) # Goodness of fit for each treated over pre-treatment period, used in the original paper
sigma_cutoff = mean(sigma) # for later use: correction during Fisher test


## 4.2 Non-Penalized Synthetic Control
NPsol = estval$Wsol[1,,]
colnames(NPsol) = rownames(X[d==0,])
phiNP = apply((y[TestPeriod,d==1] - y[TestPeriod,d==0]%*%t(NPsol)),1,mean)

sigma = sqrt(apply((X1 - X0%*%t(NPsol))^2,2,mean))
sigma_cutoffNP = mean(sigma) 

# Number of active controls
apply(NPsol>0,1,sum)
print("mean nb. active control units"); mean(apply(NPsol>0,1,sum))


### 5. Fisher Test of No-Effect Assumption (C=0)
set.seed(1207990)
R = 5000 # Number of replications
alpha = sqrt(3) # correction cut-off (see original paper)
lambda.set = c(seq(0,0.1,.01),seq(.2,1.5,.1)) # sequence of lambdas to test
ResultP = matrix(nrow=R, ncol=length(TestPeriod))
ResultP_C = matrix(nrow=R, ncol=length(TestPeriod))
ResultNP = matrix(nrow=R, ncol=length(TestPeriod))
ResultNP_C = matrix(nrow=R, ncol=length(TestPeriod))
t_start = Sys.time()
pb = txtProgressBar(style = 3)
for(r in 1:R){
  dstar = sample(d)
  X0star = y[PreTreatPeriod,dstar==0]; X1star = y[PreTreatPeriod,dstar==1]
  
  ### SELECTION OF LAMBDA OPT FOR THIS ITERATION ### 
  estval = regsynthpath(X0star, X1star,Y0,Y1,V,lambda.set,tol=1e-6)
  MSPE = vector(length=length(lambda.set))
  
  for(k in 1:length(lambda.set)){
    MSPE[k] = mean(apply((y[324:354,dstar==1] - y[324:354,dstar==0]%*%t(estval$Wsol[k,,]))^2,2,mean))
  }
  
  Wsolstar = estval$Wsol[which(MSPE==min(MSPE)),,] # COLLECT W(lambda.opt)
  
  # Not corrected
  ResultP[r,] = apply((y[TestPeriod,dstar==1] - y[TestPeriod,dstar==0]%*%t(Wsolstar)),1,mean)
  
  # Corrected
  sigmastar = sqrt(apply((X1star - X0star%*%t(Wsolstar))^2,2,mean))
  omegastar_C = rep(1,sum(d))
  omegastar_C[sigmastar>alpha*sigma_cutoff] = 0
  omegastar_C = omegastar_C/sum(omegastar_C)
  ResultP_C[r,] = (y[TestPeriod,dstar==1] - y[TestPeriod,dstar==0]%*%t(Wsolstar))%*%omegastar_C
  
  ### NON-PENALIZED, LAMBDA=0 ###
  NPsolstar = estval$Wsol[1,,]
  
  # Not corrected
  ResultNP[r,] = apply((y[TestPeriod,dstar==1] - y[TestPeriod,dstar==0]%*%t(NPsolstar)),1,mean)
  
  # Corrected
  sigmastar = sqrt(apply((X1star - X0star%*%t(NPsolstar))^2,2,mean))
  omegastar_C = rep(1,sum(d))
  omegastar_C[sigmastar>alpha*sigma_cutoffNP] = 0
  omegastar_C = omegastar_C/sum(omegastar_C)
  ResultNP_C[r,] = (y[TestPeriod,dstar==1] - y[TestPeriod,dstar==0]%*%t(NPsolstar))%*%omegastar_C
  
  setTxtProgressBar(pb, r/R)
}
close(pb)
print(Sys.time()-t_start)


## 5.1 Tables

# Penalized / Non-Corrected
cumphiP = cumsum(phiP[16:length(phiP)])
cumResultP = t(apply(ResultP[,16:length(phiP)],1,cumsum))
cumphi_q = t(mapply(function(t) quantile(cumResultP[,t], probs = c(.005,.025,.975,.995)), 1:ncol(cumResultP)))

TableP = data.frame("Estimate"=cumphiP,"Q"=cumphi_q)

print("Event day 0"); print(TableP[1,])
print("Event day 10"); print(TableP[11,])

# Penalized / Corrected
cumResult_C = t(apply(ResultP_C[,16:length(phiP)],1,cumsum))
cumphi_qC = t(mapply(function(t) quantile(cumResult_C[,t], probs = c(.005,.025,.975,.995)), 1:ncol(cumResult_C)))

TableP_Corrected = data.frame("Estimate"=cumphiP,"Q"=cumphi_qC)

# Non-Penalized / Non-Corrected
cumphiNP = cumsum(phiNP[16:length(phiNP)])
cumResultNP = t(apply(ResultNP[,16:length(phiNP)],1,cumsum))
cumphi_q = t(mapply(function(t) quantile(cumResultNP[,t], probs = c(.005,.025,.975,.995)), 1:ncol(cumResultNP)))

TableNP = data.frame("Estimate"=cumphiNP,"Q"=cumphi_q)

# Non-Penalized / Corrected
cumResult_C = t(apply(ResultNP_C[,16:length(phiNP)],1,cumsum))
cumphi_qC = t(mapply(function(t) quantile(cumResult_C[,t], probs = c(.005,.025,.975,.995)), 1:ncol(cumResult_C)))

TableNP_Corrected = data.frame("Estimate"=cumphiNP,"Q"=cumphi_qC)

ToPrint = t(rbind(TableP[1,],TableP[11,],TableP_Corrected[1,],TableP_Corrected[11,],
                  TableNP[1,],TableNP[11,],TableNP_Corrected[1,],TableNP_Corrected[11,]))
colnames(ToPrint) = c("Penalized, NC, Day 0","Penalized, NC, Day 10","Penalized, C, Day 0","Penalized, C, Day 10",
                      "Non-Penalized, NC, Day 0","Non-Penalized, NC, Day 10","Non-Penalized, C, Day 0","Non-Penalized, C, Day 10")
stargazer(t(ToPrint))

fileConn = file("plot/GeithnerResultTable.txt")
writeLines(stargazer(t(ToPrint)), fileConn)
close(fileConn)

### A. Not corrected

# Compute .025 and .975 quantiles of CAR for each date
phi_q = t(mapply(function(t) quantile(ResultP[,t], probs = c(.005,.025,.975,.995)), 1:length(TestPeriod)))
ATTdata = ts(cbind(phi_q[,1:2],phiP,phi_q[,3:4]),start=c(-15), freq=1)

### Figure 2: Geithner connected firms effect vs. random permutations (Currently in paper)
pdf("plot/GeithnerAR_FisherTest.pdf", width=10, height=6)
plot(ATTdata, plot.type="single",
     col=c("firebrick","firebrick","firebrick","firebrick","firebrick"), lwd=c(1,1,2,1,1),
     lty=c(3,4,1,4,3),xlab="Day", ylab="AR, in pp",
     ylim=c(-.15,.15))
abline(h=0,
       lty=2,col="grey")
lim <- par("usr")
rect(0, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(-15,-.075,
       legend=c("Estimate", ".95 confidence bands of Fisher distrib.",".99 confidence bands of Fisher distrib."),
       col=c("firebrick","firebrick"), lwd=c(2,1,1),
       lty=c(1,4,3))
dev.off()


### 6. .95 Confidence interval for CAR[0] and CAR[10]
GeiCI0 = conf.interval.Geithner(d,as.matrix(y[GeiNomDate,]),t(y[PreTreatPeriod,]),V,lambda=lambda.opt.MSPE,B=5000,alpha=.05)
fileConn = file("plot/outputCI0.txt")
writeLines(paste(GeiCI0$c.int), fileConn)
close(fileConn)

GeiCI10 = conf.interval.Geithner(d,t(y[GeiNomDate:(GeiNomDate+10),]),t(y[PreTreatPeriod,]),V,lambda=lambda.opt.MSPE,B=5000,alpha=.05)
fileConn = file("plot/outputCI10.txt")
writeLines(paste(GeiCI10$c.int), fileConn)
close(fileConn)