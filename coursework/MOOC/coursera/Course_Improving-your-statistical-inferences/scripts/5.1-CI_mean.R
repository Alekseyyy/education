#This scripts will simulate a single sample, and calculate the mean
#The gold background illustrates the 95% prediction interval (PI), The orange background illustrates the 95% confidence interval
#The black dotted line illustrates the true mean. 95% of the CI should contain the true mean
#Then, a simulation is run. The simulations generates a large number of additional samples
#The simulation returns the number of CI that contain the mean, and returns the % of means from future studies that fall within the 95% of the original study
#This is known as the capture percentage. It differs from (and is lower than) the confidence interval 

if(!require(ggplot2)){install.packages('ggplot2')}
library(ggplot2)
if(!require(Rcpp)){install.packages('Rcpp')}
library(Rcpp)

n=20 #set sample size
nSims<-100000 #set number of simulations

x<-rnorm(n = n, mean = 100, sd = 15) #create sample from normal distribution

#95% Confidence Interval
CIU<-mean(x)+qt(0.975, df = n-1)*sd(x)*sqrt(1/n)
CIL<-mean(x)-qt(0.975, df = n-1)*sd(x)*sqrt(1/n)

#95% Prediction Interval
PIU<-mean(x)+qt(0.975, df = n-1)*sd(x)*sqrt(1+1/n)
PIL<-mean(x)-qt(0.975, df = n-1)*sd(x)*sqrt(1+1/n)

#plot data
#png(file="CI_mean.png",width=2000,height=2000, res = 300)
ggplot(as.data.frame(x), aes(x))  + 
  geom_rect(aes(xmin=PIL, xmax=PIU, ymin=0, ymax=Inf), fill="gold") + #draw orange CI area
  geom_rect(aes(xmin=CIL, xmax=CIU, ymin=0, ymax=Inf), fill="#E69F00") + #draw orange CI area
  geom_histogram(colour="black", fill="grey", aes(y=..density..), binwidth=2) +
  xlab("IQ") + ylab("number of people")  + ggtitle("Data") + theme_bw(base_size=20) + 
  theme(panel.grid.major.x = element_blank(), axis.text.y = element_blank(), panel.grid.minor.x = element_blank()) + 
  geom_vline(xintercept=mean(x), colour="black", linetype="dashed", size=1) + 
  coord_cartesian(xlim=c(50,150)) + scale_x_continuous(breaks=c(50,60,70,80,90,100,110,120,130,140,150)) +
  annotate("text", x = mean(x), y = 0.02, label = paste("Mean = ",round(mean(x)),"\n","SD = ",round(sd(x)),sep=""), size=6.5)
#dev.off()

#Simulate Confidence Intervals
CIU_sim<-numeric(nSims)
CIL_sim<-numeric(nSims)
mean_sim<-numeric(nSims)

for(i in 1:nSims){ #for each simulated experiment
  x<-rnorm(n = n, mean = 100, sd = 15) #create sample from normal distribution
  CIU_sim[i]<-mean(x)+qt(0.975, df = n-1)*sd(x)*sqrt(1/n)
  CIL_sim[i]<-mean(x)-qt(0.975, df = n-1)*sd(x)*sqrt(1/n)
  mean_sim[i]<-mean(x) #store means of each sample
}

#Save only those simulations where the true value was inside the 95% CI
CIU_sim<-CIU_sim[CIU_sim<100]
CIL_sim<-CIL_sim[CIL_sim>100]

cat((100*(1-(length(CIU_sim)/nSims+length(CIL_sim)/nSims))),"% of the 95% confidence intervals contained the true mean")

#Calculate how many times the observed mean fell within the 95% CI of the original study
mean_sim<-mean_sim[mean_sim>CIL&mean_sim<CIU]
cat("The capture percentage for the plotted study, or the % of values within the observed confidence interval from",CIL,"to",CIU,"is:",100*length(mean_sim)/nSims,"%")

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/