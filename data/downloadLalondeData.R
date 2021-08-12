#' Script to download Lalonde (1986) dataset
#' from the causalsens package
#' 12/08/2021
#' @author : jeremylhour

install.packages("causalsens", repos = "http://cran.us.r-project.org")
require("causalsens")

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

# Form the five required matrices
X0 = X[d==0,]
X1 = X[d==1,]
Y0 = y[d==0]
Y1 = y[d==1]
X0_unscaled = X_unscaled[d==0,]

# output data
write.table(X0, file="X0.txt", row.names=FALSE, col.names=TRUE)
write.table(X1, file="X1.txt", row.names=FALSE, col.names=TRUE)
write.table(Y0, file="Y0.txt", row.names=FALSE, col.names=TRUE)
write.table(Y1, file="Y1.txt", row.names=FALSE, col.names=TRUE)
write.table(X0_unscaled, file="X0_unscaled.txt", row.names=FALSE, col.names=TRUE)