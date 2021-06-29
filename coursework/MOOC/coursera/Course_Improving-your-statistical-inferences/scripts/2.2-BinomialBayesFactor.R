H0<-0.5 #Set the point null hypothesis you want to calculate the Bayes Factor for
n<-50 #set total trials
x<-20 #set successes
aprior<-2 #Set the alpha for the Beta distribution for the prior
bprior<-2 #Set the beta for the Beta distribution for the prior

alikelihood<-x+1 #Calculate the alpha for the Beta distribution for the likelihood
blikelihood<-n-x+1 #Calculate the beta for the Beta distribution for the likelihood
aposterior<-aprior+alikelihood-1 #Calculate the alpha for the Beta distribution for the posterior
bposterior<-bprior+blikelihood-1 #Calculate the beta for the Beta distribution for the posterior

theta<-seq(0,1,0.001) #create theta range from 0 to 1
#png(file="PriorLikelihoodPosterior.png",width=3000,height=3000, res = 500)
prior <- dbeta(theta, aprior, bprior)
likelihood <- dbeta(theta, alikelihood, blikelihood)
posterior <- dbeta(theta, aposterior, bposterior)
plot(theta, posterior, ylim=c(0, 15), type = "l", lwd = 3, xlab = bquote(theta), ylab = "Density", las = 1)
lines(theta, prior, col="grey", lwd = 3)
lines(theta, likelihood, lty = 2, lwd = 3, col="dodgerblue")
BF10<-dbeta(H0, aposterior, bposterior)/dbeta(H0, aprior, bprior)
points(H0,dbeta(H0, aposterior, bposterior), pch = 19)
points(H0,dbeta(H0, aprior, bprior), pch = 19, col="grey")
segments(H0, dbeta(H0, aposterior, bposterior), H0, dbeta(H0, aprior, bprior), lty=2)
title(paste('Bayes Factor:',round(BF10,digits=2)))
#dev.off()

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/
