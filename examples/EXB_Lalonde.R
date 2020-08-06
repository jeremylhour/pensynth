#' Penalized Synthetic Control -- Empirical Applciation: Lalonde Data
#' 02/04/2020
#' @author Jeremy L'Hour

# setwd("//ulysse/users/JL.HOUR/1A_These/A. Research/RegSynthProject/regsynth")
# setwd("/Users/jeremylhour/Documents/code/pensynth")
rm(list=ls())

### Load packages
library("MASS")
library("ggplot2")
library("gtable")
library("grid")
library("reshape2")
library("LowRankQP")
library("doParallel")
library("data.table")

### Load user functions
source("functions/wsoll1.R")
source("functions/regsynth.R")
source("functions/regsynthpath.R")
source("functions/pensynth_parallel.R")
source("functions/TZero.R")
source("functions/estimator_matching.R")
source("functions/get_stats.R")

### define extra functions
mMscale <- function(X){
  X = as.matrix(X)
  mins = apply(X,2,min); maxs = apply(X,2,max)
  return(scale(X, center=mins, scale=maxs-mins))
}

Table = data.frame()

############################
############################
### 1. Experimental data ###
############################
############################

library("causalsens")
data(lalonde.exp)

d = lalonde.exp[,"treat"]
y = lalonde.exp[,"re78"]

X = data.frame(lalonde.exp[,c("age","education","married","black","hispanic","re74","re75","nodegree")],
               "NoIncome74"=as.numeric(lalonde.exp[,"re74"]==0),
               "NoIncome75"=as.numeric(lalonde.exp[,"re75"]==0)
)

# Statistics on treated
Table[1,names(X)] = round(apply(X[d==1,],2,mean),digits=2)
Table[1,"Sample_size"] = sum(d)

# Statistics on non-treated (experimental)
Table[2,names(X)] = round(apply(X[d==0,],2,mean),digits=2)
Table[2,"Sample_size"] = sum(1-d)

# Treatment effect
exp_reg = lm(y ~ d)
Table[2,"Treatment_effect"] = exp_reg$coefficients["d"]

remove(lalonde.exp,y,d,X)

############################
############################
### 2. Loading PSID data ###
############################
############################


data(lalonde.psid)

d = lalonde.psid[,"treat"]
y = lalonde.psid[,"re78"]

X = data.frame(lalonde.psid[,c("age","education","married","black","hispanic","re74","re75","nodegree")],
               "NoIncome74"=as.numeric(lalonde.psid[,"re74"]==0),
               "NoIncome75"=as.numeric(lalonde.psid[,"re75"]==0)
)

X_unscaled = X # Save unscaled data to compute statistics

# Rescale by dividing by standard error
X[,c("age","education","married","black","hispanic","nodegree","NoIncome74","NoIncome75")] = mapply(function(x) X[,x]/sd(X[d==1,x]), c("age","education","married","black","hispanic","nodegree","NoIncome74","NoIncome75"))

# Rescale income by standard error cutting outliers (above quantile 90% for the treated)
Q90 = mapply(function(x) quantile(X[d==1,x], p=.9), c("re74","re75"))
re74_trunc_std = sd(X[(d==1 & X[,"re74"] < Q90[1]),"re74"])
re75_trunc_std = sd(X[(d==1 & X[,"re75"] < Q90[2]),"re75"])

X[,"re74"] = X[,"re74"]/re74_trunc_std 
X[,"re75"] = X[,"re75"]/re75_trunc_std 

X = as.matrix(X)

X0 = t(X[d==0,]); X1 = t(X[d==1,])
Y0 = y[d==0]; Y1 = y[d==1]
V = diag(ncol(X))

# Statistics on non-treated (PSID)
Table[3,names(X_unscaled)] = round(apply(X_unscaled[d==0,],2,mean),digits=2)
Table[3,"Sample_size"] = sum(1-d)

##########################################
##########################################
### 3. Synthetic control, fixed lambda ###
##########################################
##########################################


# For synthetic control: eliminate untreated rows with similar value of X (just keep one) 
# and assign average value of the outcome

keys = c('age', 'education',  'married', 'black', 'hispanic', 're74', 're75', 'nodegree', 'NoIncome74', 'NoIncome75')
X0_unique = as.data.table(cbind(Y0,t(X0)))
X0_unique = X0_unique[,list(Y0_average = mean(Y0)), keys]
Y0_average = as.vector(X0_unique[,Y0_average])
X0_unique = t(as.matrix(X0_unique[,..keys]))

# lambda = .1
t_start <- Sys.time()
sol = regsynth(X0_unique,X1,Y0_average,Y1,V,pen=.1,parallel=TRUE)
print(Sys.time()-t_start)

### A MODIFIER ENSUITE POUR AVOIR LES BONNES STATS

X0_unscaled = X_unscaled[d==0,]
X0_unscaled_unique = as.data.table(cbind(Y0,X0_unscaled))
X0_unscaled_unique = X0_unscaled_unique[,list(Y0_average = mean(Y0)), keys]
X0_unscaled_unique = t(as.matrix(X0_unscaled_unique[,..keys]))

# Statistics on fixed lambda
Table[4,"lambda"] = .1
Table[4,names(X_unscaled)] = round(apply(X0_unscaled_unique%*%t(sol$Wsol),1,mean), digits=2)
Table[4,"Treatment_effect"] = sol$ATT

sparsity_index = apply(sol$Wsol>0,1,sum)
Table[4,"Min_density"] = min(sparsity_index)
Table[4,"Median_density"] = median(sparsity_index)
Table[4,"Max_density"] = max(sparsity_index)

activ_index = apply(sol$Wsol>0,2,sum)
Table[4,"Sample_size"] = sum(activ_index>0)


df = data.frame(weight=sol$CATT)
ggplot(df, aes(x=weight)) + geom_histogram(color="black", fill="lightblue") +
  geom_vline(aes(xintercept=mean(weight)),color="red", linetype="dashed", size=1) +
  labs(title="",x="Treatment effect (dollars)", y = "")

############################################
############################################
### 3. Synthetic control, optimal lambda ###
############################################
############################################

### To limit computational time:
### select approx. 185 control units that resembles the treated
### For each one of them, creates its synthetic control using all the other control units
### Optimize lambda


W_matching = matchest(X0_unique,X1,m=4)$Wsol
X0_matched = which(apply(W_matching>0,2,sum)>0)
length(X0_matched) # This gives exactly 170 control units

# Setting up the procedure
lambda = c(0,.00001,.01,.1,.15,seq(.25,5,.1)) # set of lambda to be considered for optim
set.seed(12071990)

# B. lambda = lambdaopt
keep_tau = matrix(nrow=length(lambda), ncol=length(X0_matched))
for(k in 1:length(X0_matched)){
  print(paste("Creating penalized synth for control unit",k,"of", length(X0_matched)))
  X1k = as.matrix(X0_unique[,X0_matched[k]])
  X0k = as.matrix(X0_unique[,-X0_matched[k]])
  Y1k = Y0_average[X0_matched[k]]
  Y0k = Y0_average[-X0_matched[k]]
  solpath = pensynth_parallel(X0k,X1k,Y0k,Y1k,lambda=lambda)
  keep_tau[,k] = solpath$CATT
}

# The one that optimizes RMSE
curve_RMSE = sqrt(apply(keep_tau^2,1,mean))
lambda_opt_RMSE = min(lambda[which(curve_RMSE==min(curve_RMSE))])
print(paste("RMSE optimal lambda:",lambda_opt_RMSE))
sol_RMSE = regsynth(X0_unique,X1,Y0_average,Y1,V,pen=lambda_opt_RMSE,parallel=TRUE)
Wsol_opt_RMSE = sol_RMSE$Wsol

# Statistics on RMSE-opt lambda
Table[5,"lambda"] = lambda_opt_RMSE
Table[5,names(X_unscaled)] = round(apply(X0_unscaled_unique%*%t(sol_RMSE$Wsol),1,mean), digits=2)
Table[5,"Treatment_effect"] = sol_RMSE$ATT

sparsity_index = apply(sol_RMSE$Wsol>0,1,sum)
Table[5,"Min_density"] = min(sparsity_index)
Table[5,"Median_density"] = median(sparsity_index)
Table[5,"Max_density"] = max(sparsity_index)

activ_index = apply(sol_RMSE$Wsol>0,2,sum)
Table[5,"Sample_size"] = sum(activ_index>0)

# The one that optimizes bias (if different)
curve_bias = abs(apply(keep_tau,1,mean))
lambda_opt_bias = min(lambda[which(curve_bias==min(curve_bias))])
print(paste("bias optimal lambda:",lambda_opt_bias))

if(lambda_opt_bias != lambda_opt_RMSE){
  sol_bias = regsynth(X0_unique,X1,Y0_average,Y1,V,pen=lambda_opt_bias,parallel=TRUE)
  Wsol_opt_bias = sol_bias$Wsol
} else {
  sol_bias = sol_RMSE
  Wsol_opt_bias = Wsol_opt_RMSE
}

# Statistics on bias-opt lambda
Table[6,"lambda"] = lambda_opt_bias
Table[6,names(X_unscaled)] = round(apply(X0_unscaled_unique%*%t(sol_bias$Wsol),1,mean), digits=2)
Table[6,"Treatment_effect"] = sol_bias$ATT

sparsity_index = apply(sol_bias$Wsol>0,1,sum)
Table[6,"Min_density"] = min(sparsity_index)
Table[6,"Median_density"] = median(sparsity_index)
Table[6,"Max_density"] = max(sparsity_index)

activ_index = apply(sol_bias$Wsol>0,2,sum)
Table[6,"Sample_size"] = sum(activ_index>0)


########################
########################
### 4. Matching, 1NN ###
########################
########################

sol_1NN = matchest(X0,X1,Y0,Y1,m=1)

# Statistics on 1-NN
Table[7,names(X_unscaled)] = round(apply(t(X_unscaled[d==0,])%*%t(sol_1NN$Wsol),1,mean),digits=2)
Table[7,"Treatment_effect"] = sol_1NN$ATT

sparsity_index = apply(sol_1NN$Wsol>0,1,sum)
Table[7,"Min_density"] = min(sparsity_index)
Table[7,"Median_density"] = median(sparsity_index)
Table[7,"Max_density"] = max(sparsity_index)

activ_index = apply(sol_1NN$Wsol>0,2,sum)
Table[7,"Sample_size"] = sum(activ_index>0)

############################
############################
### 5. Matching, opt. NN ###
############################
############################

# Setting up the procedure
M = 1:30 # set of lambda to be considered for optim
set.seed(12071990)

# B. lambda = lambdaopt
keep_tau_NN = matrix(nrow=length(M), ncol=length(X0_matched))
for(k in 1:length(X0_matched)){
  X1k = as.matrix(X0[,X0_matched[k]])
  X0k = as.matrix(X0[,-X0_matched[k]])
  Y1k = Y0[X0_matched[k]]
  Y0k = Y0[-X0_matched[k]]
  print(paste("Creating nearest neighbor for control unit",k,"of", length(X0_matched)))
  for(m_NN in 1:length(M)){
    sol_NN = matchest(X0k,X1k,Y0k,Y1k,m=m_NN)
    keep_tau_NN[m_NN,k] = sol_NN$CATT 
  }
}

# The one that optimizes RMSE
curve_RMSE_NN = sqrt(apply(keep_tau_NN^2,1,mean))
M_opt_RMSE_NN = min(M[which(curve_RMSE_NN==min(curve_RMSE_NN))])
print(paste("RMSE optimal m:",M_opt_RMSE_NN))
sol_RMSE_NN = matchest(X0,X1,Y0,Y1,m=M_opt_RMSE_NN)


# Statistics on RMSE-opt NN
Table[8,names(X_unscaled)] = round(apply(t(X_unscaled[d==0,])%*%t(sol_RMSE_NN$Wsol),1,mean),digits=2)
Table[8,"Treatment_effect"] = sol_RMSE_NN$ATT

sparsity_index = apply(sol_RMSE_NN$Wsol>0,1,sum)
Table[8,"Min_density"] = min(sparsity_index)
Table[8,"Median_density"] = median(sparsity_index)
Table[8,"Max_density"] = max(sparsity_index)

activ_index = apply(sol_RMSE_NN$Wsol>0,2,sum)
Table[8,"Sample_size"] = sum(activ_index>0)


### Adding labels
rownames(Table) = c("Treated", "Experimental", "PSID",
                    "PenSynth fixed lambda", "PenSynth MSE opt lambda","PenSynth bias opt lambda",
                    "Matching 1NN", "Matching opt NN")

print(t(Table))

###########################################
###########################################
### 6. Inference with Synthetic control ###
###########################################
###########################################

# Lambda = .1
B = 100
set.seed(12071990)

indiv_TE = matrix(nrow=B, ncol=ncol(X1))

for(b in 1:B){
  print(paste("Permutation",b,"of", B))
  # Draw new sample
  d_tilde = sample(d)
  X0_tilde = t(X[d_tilde==0,]); X1_tilde = t(X[d_tilde==1,])
  Y0_tilde = y[d_tilde==0]; Y1_tilde = y[d_tilde==1]
  
  # Compute synthetic control solution
  sol_tilde = regsynth(X0_tilde,X1_tilde,Y0_tilde,Y1_tilde,V,pen=.1,parallel=TRUE)
  indiv_TE[b,] = sol_tilde$CATT
}

p_value <- function(indiv_TE, method="sumrank"){
  if(method=="sumrank"){
    test_stat = apply(matrix(rank(indiv_TE),nrow=nrow(indiv_TE),ncol=ncol(indiv_TE)),1,sum)
  } else if(method=="aggregate"){
    test_stat = apply(indiv_TE,1,mean)
  }
  return(mean(test_stat>=test_stat[1]))
}

# Computing p-values
p_value(rbind(t(sol$CATT),indiv_TE)) # sum of ranks
p_value(rbind(t(sol$CATT),indiv_TE),method="aggregate") # aggregate treatment effect

# Sum of rank
test_stat_SR = apply(matrix(rank(rbind(t(sol$CATT),indiv_TE)),nrow=nrow(indiv_TE)+1,ncol=ncol(indiv_TE)),1,sum)

df = data.frame(stat=test_stat_SR)
plot1 = ggplot(df, aes(x=stat)) + geom_histogram(color="black", fill="lightblue") +
  geom_vline(aes(xintercept=test_stat_SR[1]),color="red", linetype="dashed", size=1) +
  labs(title="",x="Sum of ranks", y = "")

# abs(mean treatment effect)
test_stat = apply(rbind(t(sol$CATT),indiv_TE),1,mean)

df = data.frame(stat=test_stat)
plot2 = ggplot(df, aes(x=stat)) + geom_histogram(color="black", fill="lightblue") +
  geom_vline(aes(xintercept=test_stat[1]),color="red", linetype="dashed", size=1) +
  labs(title="",x="Aggregate Treatment Effect", y = "")

require(gridExtra)

pdf("plot/Lalonde_test_100.pdf",width = 12, height=5)
grid.arrange(plot1, plot2, ncol=2)
dev.off()

save.image(file = 'rsessions/Lalonde_Example.RData')