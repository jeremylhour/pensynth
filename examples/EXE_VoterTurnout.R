### Application from Yiqing Xu's paper
### "Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models"
### 10/07/2019
### Jeremy L'Hour

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
library("gsynth")
library("foreach")
library("doParallel")
library("geometry")

### Load user functions
source("functions/wsoll1.R")
source("functions/regsynth.R")
source("functions/regsynthpath.R")
source("functions/TZero.R")
source("functions/synthObj.R")
source("functions/perm.test.R")
source("functions/conf.interval.R")
source("functions/bias.R")

### 0. Load Data
data("gsynth")

### Switch to wide dataset where 1 row = 1 state
wide_turnout = reshape(turnout[,c("abb","year","turnout")], idvar = "abb", timevar = "year", direction = "wide")
wide_EDR = reshape(turnout[,c("abb","year","policy_edr")], idvar = "abb", timevar = "year", direction = "wide")

# Control and Treated
StateID = unique(turnout[,"abb"])
Treated = unique(turnout[turnout[,"policy_edr"]==1,"abb"])
Control = StateID[!(StateID %in% Treated)]
D = (wide_EDR[,"abb"] %in% Treated)

# Year of Treatment
Treated.1976 = wide_EDR[wide_EDR[,"policy_edr.1976"] == 1,"abb"]
D.1976 = (wide_EDR[,"abb"] %in% Treated.1976)
Treated.1996 = wide_EDR[wide_EDR[,"policy_edr.1996"] == 1,"abb"]
D.1996 = (wide_EDR[,"abb"] %in% Treated.1996)

# Outcomes
Outcomes = t(wide_turnout[,-1])
colnames(Outcomes) = StateID
PreTreatment = 1:14
Training = 1:8
Validation = 9:14
Treatment = 15:24

# Forming matrices
X1 = Outcomes[PreTreatment,D==1]; Y1 = Outcomes[Treatment,D==1]; X1train = Outcomes[Training,D==1]; X1val = Outcomes[Validation,D==1];
X0 = Outcomes[PreTreatment,D==0]; Y0 = Outcomes[Treatment,D==0]; X0train = Outcomes[Training,D==0]; X0val = Outcomes[Validation,D==0];
V = diag(length(Training))


###############################
###############################
### I. ALL TREATED TOGETHER ###
###############################
###############################

### 1. Optimal lambda
lambda = c(seq(0,.5,.001),seq(.5,2,.025)) # sequence of lambdas to test
estval = regsynthpath(X0train,X1train,as.matrix(Y0[1,]),as.matrix(Y1[1,]),V,lambda,tol=1e-6)
MSPE = vector(length=length(lambda))

for(k in 1:length(lambda)){
  MSPE[k] = mean(apply((X1val - X0val%*%t(estval$Wsol[k,,]))^2,2,mean))
}
lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))]) 

### Figure 1: MSPE
matplot(lambda, MSPE, type="b", pch=20, lwd=1,
        main=expression("MSPE, "*lambda^{opt}*"= .025."), col="steelblue",
        xlab=expression(lambda), ylab="MSPE")
abline(v=lambda.opt.MSPE,lty=2,lwd=2,col="grey")

### 2. Estimation

## 2.1 Penalized Synthetic Control
Psol =  regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),lambda.opt.MSPE,tol=1e-6)$Wsol
colnames(Psol) = colnames(X0)
rownames(Psol) = colnames(X1)

# Number of active controls
apply(Psol>0,1,sum)
print("mean nb. active control units"); mean(apply(Psol>0,1,sum))

# Estimate
tauP = apply((Y1 - Y0%*%t(Psol)),1,mean)
tauP_bc = tauP - apply(bias(X0,X1,Outcomes[Treatment,],D,Psol),1,mean) # bias corrected

MSPE_RatioP = sum(tauP^2) / sum(apply((X1 - X0%*%t(Psol)),1,mean)^2)
MSPE_RatioP_bc = sum(tauP_bc^2) / sum(apply((X1 - X0%*%t(Psol)),1,mean)^2)

## 2.2 Non-Penalized Synthetic Control
NPsol = regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),0,tol=1e-6)$Wsol
colnames(NPsol) = colnames(X0)
rownames(NPsol) = colnames(X1)

# Number of active controls
apply(NPsol>0,1,sum)
print("mean nb. active control units"); mean(apply(NPsol>0,1,sum))

# Estimate
tauNP = apply((Y1 - Y0%*%t(NPsol)),1,mean)
tauNP_bc = tauNP - apply(bias(X0,X1,Outcomes[Treatment,],D,NPsol),1,mean) # bias corrected

MSPE_RatioNP = sum(tauNP^2) / sum(apply((X1 - X0%*%t(NPsol)),1,mean)^2)
MSPE_RatioNP_bc = sum(tauNP_bc^2) / sum(apply((X1 - X0%*%t(NPsol)),1,mean)^2)

Estimate = data.frame(rbind(tauP,tauP_bc,tauNP,tauNP_bc))
Estimate[,"MSPE Ratio"] = c(MSPE_RatioP,MSPE_RatioP_bc,MSPE_RatioNP,MSPE_RatioNP_bc)
rownames(Estimate) = c("Pen. SC", "Pen. SC (BC)", "SC", "SC (BC)")

### 2bis. Connections in Augmented Delaunay Triangulation
# + Non-Penalized Synthetic Control Using Only the Neighbors
n1 = ncol(X1); n0 = ncol(X0)
Nmatrix = matrix(0,nrow = n1, ncol = n0)
pure.sol = matrix(0,nrow=n1,ncol=n0)

for(i in 1:n1){
  print(paste("Iteration ",i))
  AugX = cbind(X1[,i],X0)
  DT = delaunayn(t(AugX))
  index = as.logical(apply(DT == 1,1,sum))
  neighbors = unique(c(DT[index,])) - 1
  neighbors = sort(neighbors[neighbors>0])
  Nmatrix[i,neighbors] = 1
  
  sol = wsoll1(X0[,neighbors],X1[,i],diag(nrow(X1)),0)
  sol = TZero(sol)
  pure.sol[i,sort(neighbors)] = sol
}

colnames(Nmatrix) = colnames(X0)
rownames(Nmatrix) = colnames(X1)
colnames(pure.sol) = colnames(X0)
rownames(pure.sol) = colnames(X1)

tauPure = apply((Y1 - Y0%*%t(pure.sol)),1,mean)
# NB: does not seem to work very and does not agree with Theorem 3 in our paper (computation problem? coding error?)
Smatrix = 1*(NPsol > 0)

(Nmatrix>0)*(Smatrix)

### 3. Inference (Fisher Test of No Effect)
set.seed(1207990)
R = 10000 # Number of replications
lambda.set = seq(0,.5,.001)

cores=detectCores()
cl = makeCluster(3) #not to overload your computer
registerDoParallel(cl)


t_start = Sys.time()
Res_PAR <- foreach(r = 1:R,.packages='LowRankQP',.combine='comb', .multicombine=TRUE) %dopar% {
  Dstar = sample(D)
  X0star = Outcomes[Training,Dstar==0]; X1star = Outcomes[Training,Dstar==1]
  Vstar = diag(nrow(X0star))
  
  ### SELECTION OF LAMBDA OPT FOR THIS ITERATION ### 
  estval = regsynthpath(X0star, X1star,as.matrix(Y0[1,]),as.matrix(Y1[1,]),V,lambda.set,tol=1e-6)
  MSPE = vector(length=length(lambda.set))
  
  for(k in 1:length(lambda.set)){
    MSPE[k] = mean(apply((Outcomes[Validation,Dstar==1] - Outcomes[Validation,Dstar==0]%*%t(estval$Wsol[k,,]))^2,2,mean))
  }
  
  lambda.opt.star = lambda.set[which(MSPE==min(MSPE))]
  
  # Penalized
  Wsolstar = regsynth(Outcomes[PreTreatment,Dstar==0],Outcomes[PreTreatment,Dstar==1],as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),lambda.opt.star,tol=1e-6)$Wsol
  
  tau_star = apply((Outcomes[Treatment,Dstar==1] - Outcomes[Treatment,Dstar==0]%*%t(Wsolstar)),1,mean)
  tau_pre_star = apply((Outcomes[PreTreatment,Dstar==1] - Outcomes[PreTreatment,Dstar==0]%*%t(Wsolstar)),1,mean)
  ResultP = sum(tau_star^2) / sum(tau_pre_star^2)
  
  # Bias correction
  tau_star = tau_star - apply(bias(Outcomes[PreTreatment,Dstar==0],Outcomes[PreTreatment,Dstar==1],Outcomes[Treatment,],Dstar,Wsolstar),1,mean)
  ResultP_BC = sum(tau_star^2) / sum(tau_pre_star^2)
  
  # Aggregated Individual Treatment Effect
  IndivTEP = apply((Outcomes[Treatment,Dstar==1] - Outcomes[Treatment,Dstar==0]%*%t(Wsolstar)),2,mean)
  
  ### NON-PENALIZED, LAMBDA=0 ###
  NPsolstar =  regsynth(Outcomes[PreTreatment,Dstar==0],Outcomes[PreTreatment,Dstar==1],as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),0,tol=1e-6)$Wsol
  
  tau_star = apply((Outcomes[Treatment,Dstar==1] - Outcomes[Treatment,Dstar==0]%*%t(NPsolstar)),1,mean)
  tau_pre_star = apply((Outcomes[PreTreatment,Dstar==1] - Outcomes[PreTreatment,Dstar==0]%*%t(NPsolstar)),1,mean)
  ResultNP = sum(tau_star^2) / sum(tau_pre_star^2)
  
  # Bias correction
  tau_star = tau_star - apply(bias(Outcomes[PreTreatment,Dstar==0],Outcomes[PreTreatment,Dstar==1],Outcomes[Treatment,],Dstar,NPsolstar),1,mean)
  ResultNP_BC = sum(tau_star^2) / sum(tau_pre_star^2)
  
  # Aggregated Individual Treatment Effect
  IndivTENP = apply((Outcomes[Treatment,Dstar==1] - Outcomes[Treatment,Dstar==0]%*%t(NPsolstar)),2,mean)
  
  list(list(ResultP),list(ResultP_BC),list(IndivTEP),list(ResultNP),list(ResultNP_BC),list(IndivTENP))
}
print(Sys.time()-t_start)
stopCluster(cl)

ResultP = t(matrix(unlist(Res_PAR[[1]]),ncol=R))
ResultP_BC = t(matrix(unlist(Res_PAR[[2]]),ncol=R))
IndivTEP = t(matrix(unlist(Res_PAR[[3]]),ncol=R))
ResultNP = t(matrix(unlist(Res_PAR[[4]]),ncol=R))
ResultNP_BC = t(matrix(unlist(Res_PAR[[5]]),ncol=R))
IndivTENP = t(matrix(unlist(Res_PAR[[6]]),ncol=R))

p.val = function(x) sum(x>=x[1])/length(x)

# PLOTS
get.plot <- function(data,obsTE,title="A Title"){
  plot_res <- ggplot(data, aes(x=V1)) + 
    geom_histogram(binwidth = 0.5, alpha=.5, position='identity',fill="steelblue", aes(y = ..density..)) +
    geom_vline(xintercept = obsTE, color="red", size=1) +
    scale_x_continuous(name="Test Statistics") +
    ggtitle(title) + 
    theme(plot.title = element_text(lineheight=.8, face="bold"),legend.position="none")
  
  return(plot_res)
}
# A. Post-Pre MSPE ratios
get.plot(as.data.frame(ResultP),Estimate[1,"MSPE Ratio"],title="Fisher Test, MSPE Ratio")
get.plot(as.data.frame(ResultP_BC),Estimate[2,"MSPE Ratio"],title="Fisher Test, MSPE Ratio")
get.plot(as.data.frame(ResultNP),Estimate[3,"MSPE Ratio"],title="Fisher Test, MSPE Ratio")
get.plot(as.data.frame(ResultNP_BC),Estimate[4,"MSPE Ratio"],title="Fisher Test, MSPE Ratio")

Estimate[,"MSPE Ratio, p-val"] = c(p.val(c(Estimate[1,"MSPE Ratio"],ResultP)),
                                   p.val(c(Estimate[2,"MSPE Ratio"],ResultP_BC)),
                                   p.val(c(Estimate[3,"MSPE Ratio"],ResultNP)),
                                   p.val(c(Estimate[4,"MSPE Ratio"],ResultNP_BC)))
                                   

# Compute Test Statistics Based on Sum of Ranks
# 1. Penalized SC
Obs_IndivTEP = apply((Y1 - Y0%*%t(Psol)),2,mean)
SumRanks = apply(matrix(rank(rbind(Obs_IndivTEP, IndivTEP)), ncol=9),1,sum)/(9*R)
p.val(SumRanks)

get.plot(data.frame("V1"=SumRanks[-1]),SumRanks[1],title="Fisher Test, Sum Ranks")

# 2. Non-Penalized SC
Obs_IndivTENP = apply((Y1 - Y0%*%t(NPsol)),2,mean)
SumRanksNP = apply(matrix(rank(rbind(Obs_IndivTENP, IndivTENP)), ncol=9),1,sum)/(9*R)
p.val(SumRanksNP)

get.plot(data.frame("V1"=SumRanksNP[-1]),SumRanksNP[1],title="Fisher Test, Sum Ranks")

Estimate[,"Sum Ranks, p-val"] = c(p.val(SumRanks),NA,p.val(SumRanksNP),NA)


### 4. Confidence Intervals

### A. Penalized Synthetic Control
CI.P = matrix(nrow=length(Treatment), ncol=2)
i=0
for(t in Treatment){
  i=i+1
  PW.CI = conf.interval(D,Outcomes[t,],t(Outcomes[PreTreatment,]),diag(length(PreTreatment)),lambda.opt.MSPE,B=10000,alpha=.05)
  CI.P[i,] = PW.CI$c.int
}


### B. Penalized Synthetic Control Plot
Y1 = apply(Outcomes[,D==1],1,mean)
Y0 = apply(Outcomes[,D==0]%*%t(Psol),1,mean)
Y0.CI = matrix(NA,nrow=length(Y1),ncol=2)
Y0.CI[Treatment,] = cbind(Y1[Treatment],Y1[Treatment]) - CI.P
Y0.pure = apply(Outcomes[,D==0]%*%t(pure.sol),1,mean)
Y0.NP = apply(Outcomes[,D==0]%*%t(NPsol),1,mean)

plotdata = ts(cbind(Y1,Y0,Y0.CI,apply(Outcomes[,D==0],1,mean)),start=c(1920), freq=1/4)

pdf("plot/TurnoutSynthetic.pdf", width=10, height=10)
plot(plotdata, plot.type="single",
     col=c("black","firebrick","firebrick","firebrick","purple"), lwd=c(2,2,1,1,2),
     lty=c(1,6,3,3,2),xlab="", ylab="Turnout %",
     ylim=c(40,85))
lim <- par("usr")
rect(1976, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(1920,85,
       legend=c("Treated", "Pen. Synthetic Control", ".95 Confidence Interval","Other States"),
       col=c("black","firebrick","firebrick","purple"), lwd=2,
       lty=c(1,6,3,2))
dev.off()


##############################
##############################
### II. BREAKDOWN BY WAVES ###
##############################
##############################





########################################################## A. 1976 #################################################################################


# Outcomes
Outcomes1976 = Outcomes[,(D==0 | D.1976==1)]
D.W1 = D.1976[(D==0 | D.1976==1)]
colnames(Outcomes1976) = StateID[(D==0 | D.1976==1)]
PreTreatment = 1:14
Training = 1:8
Validation = 9:14
Treatment = 15:24

# Forming matrices
X1 = Outcomes1976[PreTreatment,D.W1==1]; Y1 = Outcomes1976[Treatment,D.W1==1]; X1train = Outcomes1976[Training,D.W1==1]; X1val = Outcomes1976[Validation,D.W1==1];
X0 = Outcomes1976[PreTreatment,D.W1==0]; Y0 = Outcomes1976[Treatment,D.W1==0]; X0train = Outcomes1976[Training,D.W1==0]; X0val = Outcomes1976[Validation,D.W1==0];
V = diag(length(Training))

### 1. Optimal lambda
lambda = c(seq(0,.5,.001),seq(.5,2,.025)) # sequence of lambdas to test
estval = regsynthpath(X0train,X1train,as.matrix(Y0[1,]),as.matrix(Y1[1,]),V,lambda,tol=1e-6)
MSPE = vector(length=length(lambda))

for(k in 1:length(lambda)){
  MSPE[k] = mean(apply((X1val - X0val%*%t(estval$Wsol[k,,]))^2,2,mean))
}
lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))]) 

### 2. Estimation

## 2.1 Penalized Synthetic Control
Psol =  regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),lambda.opt.MSPE,tol=1e-6)$Wsol
colnames(Psol) = colnames(X0)
rownames(Psol) = colnames(X1)

# Number of active controls
apply(Psol>0,1,sum)
print("mean nb. active control units"); mean(apply(Psol>0,1,sum))

# Estimate
tauP = apply((Y1 - Y0%*%t(Psol)),1,mean)
tauP_bc = tauP - apply(bias(X0,X1,Outcomes1976[Treatment,],D.W1,Psol),1,mean) # bias corrected


### A. Penalized Synthetic Control
CI.P = matrix(nrow=length(Treatment), ncol=2)
i=0
for(t in Treatment){
  i=i+1
  PW.CI = conf.interval(D.W1,Outcomes1976[t,],t(Outcomes1976[PreTreatment,]),diag(length(PreTreatment)),lambda.opt.MSPE,B=10000,alpha=.05)
  CI.P[i,] = PW.CI$c.int
}


### B. Penalized Synthetic Control Plot
Y1 = apply(Outcomes1976[,D.W1==1],1,mean)
Y0 = apply(Outcomes1976[,D.W1==0]%*%t(Psol),1,mean)
Y0.CI = matrix(NA,nrow=length(Y1),ncol=2)
Y0.CI[Treatment,] = cbind(Y1[Treatment],Y1[Treatment]) - CI.P

plotdata = ts(cbind(Y1,Y0,Y0.CI),start=c(1920), freq=1/4)

pdf("plot/TurnoutSynthetic_Wave1.pdf", width=10, height=10)
plot(plotdata, plot.type="single",
     col=c("black","firebrick","firebrick","firebrick"), lwd=c(2,2,1,1),
     lty=c(1,6,3,3),xlab="", ylab="Turnout %",
     ylim=c(40,85))
lim <- par("usr")
rect(1976, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(1920,85,
       legend=c("Wave 1 (ME, MN, WI)", "Pen. Synthetic Control", ".95 Confidence Interval"),
       col=c("black","firebrick","firebrick"), lwd=2,
       lty=c(1,6,3))
dev.off()


########################################################## B. 1996 #################################################################################

# Outcomes
Treated1996Only = D.1996 - D.1976
Outcomes1996 = Outcomes[,(D==0 | Treated1996Only==1)]
D.W2 = D[(D==0 | Treated1996Only==1)]
colnames(Outcomes1976) = StateID[(D==0 | Treated1996Only==1)]
PreTreatment = 1:19
Training = 1:15
Validation = 16:19
Treatment = 20:24

# Forming matrices
X1 = Outcomes1996[PreTreatment,D.W2==1]; Y1 = Outcomes1996[Treatment,D.W2==1]; X1train = Outcomes1996[Training,D.W2==1]; X1val = Outcomes1996[Validation,D.W2==1];
X0 = Outcomes1996[PreTreatment,D.W2==0]; Y0 = Outcomes1996[Treatment,D.W2==0]; X0train = Outcomes1996[Training,D.W2==0]; X0val = Outcomes1996[Validation,D.W2==0];
V = diag(length(Training))

### 1. Optimal lambda
lambda = c(seq(0,.5,.001),seq(.5,2,.025)) # sequence of lambdas to test
estval = regsynthpath(X0train,X1train,as.matrix(Y0[1,]),as.matrix(Y1[1,]),V,lambda,tol=1e-6)
MSPE = vector(length=length(lambda))

for(k in 1:length(lambda)){
  MSPE[k] = mean(apply((X1val - X0val%*%t(estval$Wsol[k,,]))^2,2,mean))
}
lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))]) 

### 2. Estimation

## 2.1 Penalized Synthetic Control
Psol =  regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),lambda.opt.MSPE,tol=1e-6)$Wsol
colnames(Psol) = colnames(X0)
rownames(Psol) = colnames(X1)

# Number of active controls
apply(Psol>0,1,sum)
print("mean nb. active control units"); mean(apply(Psol>0,1,sum))

# Estimate
tauP = apply((Y1 - Y0%*%t(Psol)),1,mean)
tauP_bc = tauP - apply(bias(X0,X1,Outcomes1996[Treatment,],D.W2,Psol),1,mean) # bias corrected


### A. Penalized Synthetic Control
CI.P = matrix(nrow=length(Treatment), ncol=2)
i=0
for(t in Treatment){
  i=i+1
  PW.CI = conf.interval(D.W2,Outcomes1996[t,],t(Outcomes1996[PreTreatment,]),diag(length(PreTreatment)),lambda.opt.MSPE,B=10000,alpha=.05)
  CI.P[i,] = PW.CI$c.int
}


### B. Penalized Synthetic Control Plot
Y1 = apply(Outcomes1996[,D.W2==1],1,mean)
Y0 = apply(Outcomes1996[,D.W2==0]%*%t(Psol),1,mean)
Y0.CI = matrix(NA,nrow=length(Y1),ncol=2)
Y0.CI[Treatment,] = cbind(Y1[Treatment],Y1[Treatment]) - CI.P

plotdata = ts(cbind(Y1,Y0,Y0.CI),start=c(1920), freq=1/4)

pdf("plot/TurnoutSynthetic_Wave2.pdf", width=10, height=10)
plot(plotdata, plot.type="single",
     col=c("black","firebrick","firebrick","firebrick"), lwd=c(2,2,1,1),
     lty=c(1,6,3,3),xlab="", ylab="Turnout %",
     ylim=c(40,85))
lim <- par("usr")
rect(1996, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(1920,85,
       legend=c("Wave 2 (ID, NH, WY)", "Pen. Synthetic Control", ".95 Confidence Interval"),
       col=c("black","firebrick","firebrick"), lwd=2,
       lty=c(1,6,3))
dev.off()

########################################################## C. 2012 #################################################################################

# Outcomes
Treated2012Only = D - D.1996
Outcomes2012 = Outcomes[,(D==0 | Treated2012Only==1)]
D.W3 = D[(D==0 | Treated2012Only==1)]
colnames(Outcomes2012) = StateID[(D==0 | Treated2012Only==1)]
PreTreatment = 1:23
Training = 1:18
Validation = 19:23
Treatment = 24

# Forming matrices
X1 = Outcomes2012[PreTreatment,D.W3==1]; Y1 = t(as.matrix(Outcomes2012[Treatment,D.W3==1])); X1train = Outcomes2012[Training,D.W3==1]; X1val = Outcomes2012[Validation,D.W3==1];
X0 = Outcomes2012[PreTreatment,D.W3==0]; Y0 = t(as.matrix(Outcomes2012[Treatment,D.W3==0])); X0train = Outcomes2012[Training,D.W3==0]; X0val = Outcomes2012[Validation,D.W3==0];
V = diag(length(Training))

### 1. Optimal lambda
lambda = c(seq(0,.5,.001),seq(.5,2,.025)) # sequence of lambdas to test
estval = regsynthpath(X0train,X1train,as.matrix(Y0[1,]),as.matrix(Y1[1,]),V,lambda,tol=1e-6)
MSPE = vector(length=length(lambda))

for(k in 1:length(lambda)){
  MSPE[k] = mean(apply((X1val - X0val%*%t(estval$Wsol[k,,]))^2,2,mean))
}
lambda.opt.MSPE = min(lambda[which(MSPE==min(MSPE))]) 

### 2. Estimation

## 2.1 Penalized Synthetic Control
Psol =  regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),lambda.opt.MSPE,tol=1e-6)$Wsol
colnames(Psol) = colnames(X0)
rownames(Psol) = colnames(X1)

# Number of active controls
apply(Psol>0,1,sum)
print("mean nb. active control units"); mean(apply(Psol>0,1,sum))

# Estimate
tauP = apply((Y1 - Y0%*%t(Psol)),1,mean)
tauP_bc = tauP - apply(bias(X0,X1,Outcomes2012[Treatment,],D.W3,Psol),1,mean) # bias corrected

### 2.2 Non-Penalized Synthetic Control
NPsol =  regsynth(X0,X1,as.matrix(Y0[1,]),as.matrix(Y1[1,]),diag(nrow(X0)),0,tol=1e-6)$Wsol
colnames(NPsol) = colnames(X0)
rownames(NPsol) = colnames(X1)

### A. Penalized Synthetic Control
CI.P = matrix(nrow=length(Treatment), ncol=2)
i=0
for(t in Treatment){
  i=i+1
  PW.CI = conf.interval(D.W3,Outcomes2012[t,],t(Outcomes2012[PreTreatment,]),diag(length(PreTreatment)),lambda.opt.MSPE,B=10000,alpha=.05)
  CI.P[i,] = PW.CI$c.int
}


### B. Penalized Synthetic Control Plot
Y1 = apply(Outcomes2012[,D.W3==1],1,mean)
Y0 = apply(Outcomes2012[,D.W3==0]%*%t(Psol),1,mean)
Y0.CI = matrix(NA,nrow=length(Y1),ncol=2)
Y0.CI[Treatment,] = cbind(Y1[Treatment],Y1[Treatment]) - CI.P
Y0.CI[Treatment-1,] = cbind(Y1[Treatment],Y1[Treatment]) - CI.P

plotdata = ts(cbind(Y1,Y0,Y0.CI),start=c(1920), freq=1/4)

pdf("plot/TurnoutSynthetic_Wave3.pdf", width=10, height=10)
plot(plotdata, plot.type="single",
     col=c("black","firebrick","firebrick","firebrick"), lwd=c(2,2,1,1),
     lty=c(1,6,3,3),xlab="", ylab="Turnout %",
     ylim=c(40,85))
lim <- par("usr")
rect(2011, lim[3], lim[2], lim[4], col = rgb(0.5,0.5,0.5,1/4))
axis(1) ## add axes back
axis(2)
box() 
legend(1920,85,
       legend=c("Wave 3 (CT, IA, MT)", "Pen. Synthetic Control", ".95 Confidence Interval"),
       col=c("black","firebrick","firebrick"), lwd=2,
       lty=c(1,6,3))
dev.off()