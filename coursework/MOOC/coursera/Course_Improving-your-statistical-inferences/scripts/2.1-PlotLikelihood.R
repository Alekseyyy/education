#plot likelihood curve----
n<-10 #set total trials
x<-0 #set successes
theta<- seq(0,1,len=100) #create theta variable, from 0 to 1
like <- dbinom(x,n,theta) #create likelihood function
plot(theta,like,type='l',xlab=expression(theta), ylab='Likelihood', main="Likelihood Curve")

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/
