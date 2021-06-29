N<-100 #total number of datapoints (per condition) you are willing to collect
Looks<-5 #set number of looks at the data
nSim<-50000 #number of simulated studies
alpha<-0.05 #set alpha

D<-0 #True effect size (must be 0 when simulating Type 1 errors)

#Take care of some settings
options(scipen=100, digits=4) #disable scientific notation for numbers
LookN<-ceiling(seq(0,N,N/Looks)) #Determine at which N's to look
LookN<-LookN[-1] #remove look at 0
LookN<-LookN[LookN > 2] #Remove looks at N of 1 or 2 (not possible with t-test)
Looks<-length(LookN) #if looks are removed, change number of looks
matp<-matrix(NA, nrow=nSim, ncol=Looks) #Matrix for p-values at sequential tests
SigSeq<-numeric(Looks) #Variable to store final p-values
OptStop<-numeric(nSim) #variable to store positions of optional stopping
p<-numeric(nSim) #Variable to save optional stopping p-values

#Loop data generation for each study, then loop to perform a test for each N 
for (i in 1:nSim){
  x<-rnorm(n = N, mean = 0, sd = 1)
  y<-rnorm(n = N, mean = D, sd = 1)
  for (j in 1:Looks){
  matp[i,j]<-t.test(x[1:LookN[j]],y[1:LookN[j]], var.equal=TRUE)$p.value #perform the t-test, store
  }
  cat('Loop', i, 'of', nSim,'\n')
}

#Save Type 1 error rate for each look
for (i in 1:Looks){
  SigSeq[i] <- sum(matp[,i]<alpha)
}

#Get the positions at which are stopped, and then these p-values
for (i in 1:nSim){
  OptStop[i] <- min(which(matp[i,]<alpha))
}
OptStop[is.infinite(OptStop)] <- Looks #If nothing significant, take last p-value (fixes error warning)
for (i in 1:nSim){
  p[i] <- matp[i,OptStop[i]]
}

breaks<-100
hist(p, breaks=breaks,col="grey")
abline(h=nSim/breaks, col = "red", lty=3)

#Return Type 1 error rates for each look, and the the Type 1 error rate when only reporting the lowest p-value over all looks
cat("Type 1 error rates for look 1 to", Looks,":", SigSeq/nSim)
cat("Type 1 error rate when only the lowest p-value for all looks is reported:", sum(p<alpha)/nSim)

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/