get_stats <- function(sol,X0_unscaled,X1_unscaled){
  # Descriptive statistics
  print(paste("Average treatment effect:",sol$ATT))
  
  print(summary(sol$CATT))
  
  # Balance check: treated / untreated / synthetic
  print("Treated:")
  print(round(apply(X1_unscaled,1,mean),digits=3))
  
  print("Non-treated:")
  print(round(apply(X0_unscaled,1,mean),digits=3))
  
  print("Synthetic:")
  print(round(apply(X0_unscaled%*%t(sol$Wsol),1,mean),digits=3))
  
  # Sparsity
  sparsity_index = apply(sol$Wsol>0,1,sum)
  print(summary(sparsity_index))
  
  # Active control
  activ_index = apply(sol$Wsol>0,2,sum)
  paste("Untreated who get a positive weights at some point:",sum(activ_index>0))
  
  stats = c(round(apply(X0_unscaled%*%t(sol$Wsol),1,mean),digits=3),
            sol$ATT,
            median(sparsity_index),
            min(sparsity_index),
            max(sparsity_index),
            sum(activ_index>0)
  )
  names(stats)[11:15] = c("Treatment Effect","Median sparsity","Min. Sparsity","Max Sparsity","Active Untreated")
  return(stats)
}