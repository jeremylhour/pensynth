#' Confidence intervals for regularized Synthetic Control
#' 
#' Created: 26 octobre 2016
#' Returns R ATET estimates on reshuffled samples
#' and compute the p-value for sharp null hypothesis with constant C.
#' Used to construct confidence intervals.
#' Laste edit: 20 mars 2019
#' 
#' @param d is a vector of dimension n (treatment indicator)
#' @param X is a matrix of dimension n x p
#' @param y is a vector of dimension n (outcome)
#' @param V is a p x p matrix of weights
#' @param lambda is a positive penalty level
#' @param B is the number of draws for computing p-values
#' @param alpha is the required confidence level for the CI
#' 
#' @author Jeremy LHour

conf.interval <- function(d,y,X,V,lambda,B=10000,alpha=.05){
  # Record time 
  t_start = Sys.time()
  
  # Compute ATET on original sample
  n0 = sum(1-d); n1 = sum(d); n = n1+n0;
  X0 = t(X[d==0,]); X1 = t(X[d==1,]);
  Y0 = y[d==0]; Y1 = y[d==1];
  sol1 = regsynth(X0,X1,y[d==0],y[d==1],V,lambda)
  theta.obs = sol1$ATT
  
  print(paste("Point estimate: ",theta.obs))
  
  # Reshuffle B times the sample and get Wsol
  dpermut = replicate(B, sample(d)) # each column is a random permutation of d
  Wsol = array(dim=c(B,n1,n0))
  pb = txtProgressBar(style = 3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
  for(b in 1:B){
    X0 = t(X[dpermut[,b]==0,]); X1 = t(X[dpermut[,b]==1,]);
    solstar = regsynth(X0,X1,Y0,Y1,V,lambda)
    Wsol[b,,] = solstar$Wsol
    setTxtProgressBar(pb, b/B)
  }
  close(pb)
  
  ### Compute confidence interval based on these weights
  # Upper bound
  res0 = compute.pval(y,d,dpermut,Wsol,C=theta.obs,theta.obs)
  b = max(res0$theta.reshuffled); eps = .01
  repeat{
    res0 = compute.pval(y,d,dpermut,Wsol,C=b,theta.obs)
    if(res0$p.val < alpha) break
    res1 = compute.pval(y,d,dpermut,Wsol,C=b+eps,theta.obs)
    b = b + (alpha - res0$p.val)*eps/(res1$p.val-res0$p.val)
  }

  a = theta.obs
  f_a = 1-alpha
  repeat{
    m = (a+b)/2
    res = compute.pval(y,d,dpermut,Wsol,C=m,theta.obs)
    f_m = res$p.val - alpha
    
    if(f_m*f_a > 0){
      a = m
      res = compute.pval(y,d,dpermut,Wsol,C=a,theta.obs)
      f_a = res$p.val - alpha
    } else {
      b = m
    }
    if(abs(b-a) < .001) break
  }
  Cu = (a+b)/2
  
  # Lower bound
  res0 = compute.pval(y,d,dpermut,Wsol,C=theta.obs,theta.obs)
  a = min(res0$theta.reshuffled)
  repeat{
    res0 = compute.pval(y,d,dpermut,Wsol,C=a,theta.obs)
    if(res0$p.val < alpha) break
    res1 = compute.pval(y,d,dpermut,Wsol,C=a-eps,theta.obs)
    a = a + (alpha - res0$p.val)*eps/(res0$p.val-res1$p.val)
    print(a)
  }
  
  b = theta.obs
  f_b = 1-alpha
  repeat{
    m = (a+b)/2
    res = compute.pval(y,d,dpermut,Wsol,C=m,theta.obs)
    f_m = res$p.val - alpha
    
    if(f_m*f_b > 0){
      b = m
      res = compute.pval(y,d,dpermut,Wsol,C=b,theta.obs)
      f_b = res$p.val - alpha
    } else {
      a = m
    }
    if(abs(b-a) < .001) break
  }
  Cl = (a+b)/2

  print(paste(alpha," confidence interval: [",Cl,",",Cu,"]"))  
  print(Sys.time()-t_start)
  
  return(list(c.int=c(Cl,Cu),
              alpha=alpha,
              theta.obs=theta.obs))
}


### Auxiliary function
# compute.pval returns the value of the test statistics for each
# permutations of treatment assignment
compute.pval<- function(y,d,dpermut,Wsol,C,theta.obs){
  Ypot0 = y - d*C; Ypot1 = y + (1-d)*C;
  theta.reshuffled = mapply(function(r) mean(Ypot1[dpermut[,r]==1] - Wsol[r,,] %*% Ypot0[dpermut[,r]==0]), 1:ncol(dpermut))
  p.val = mean(abs(theta.reshuffled - C) >= abs(theta.obs-C))
  return(list(p.val=p.val,
              theta.reshuffled=theta.reshuffled))
}