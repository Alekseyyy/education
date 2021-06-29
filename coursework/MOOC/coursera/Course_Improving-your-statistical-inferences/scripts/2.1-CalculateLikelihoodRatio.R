#Calculate the likelihood ratio----

n<-280 #set total trials
x<-143 #set successes
H0 <- .5 #specify one hypothesis you want to compare with the likihood ratio
H1 <- .52 #specify another hypothesis you want to compare with the likihood ratio (you can use 1/20, or 0.05)
dbinom(x,n,H0)/dbinom(x,n,H1) #Returns the likelihood ratio of H0 over H1
dbinom(x,n,H1)/dbinom(x,n,H0) #Returns the likelihood ratio of H1 over H0

theta<- seq(0,1,len=100) #create theta variable, from 0 to 1
like <- dbinom(x,n,theta)
#png(file="LikRatio.png",width=4000,height=3000, , units = "px", res = 900)
plot(theta,like,type='l',xlab=expression(theta), ylab='Likelihood', lwd=2)
points(H0,dbinom(x,n,H0))
points(H1,dbinom(x,n,H1))
segments(H0, dbinom(x,n,H0), x/n, dbinom(x,n,H0), lty=2, lwd=2)
segments(H1, dbinom(x,n,H1), x/n, dbinom(x,n,H1), lty=2, lwd=2)
segments(x/n, dbinom(x,n,H0), x/n, dbinom(x,n,H1), lwd=2)
title(paste('Likelihood Ratio H0/H1:',round(dbinom(x,n,H0)/dbinom(x,n,H1),digits=2)," Likelihood Ratio H1/H0:",round(dbinom(x,n,H1)/dbinom(x,n,H0),digits=2)))
#dev.off()

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/
