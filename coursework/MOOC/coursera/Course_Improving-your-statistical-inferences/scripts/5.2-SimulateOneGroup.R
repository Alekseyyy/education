#load or install ggplot package
if(!require(ggplot2)){install.packages('ggplot2')}
library(ggplot2)

#Simulate one group
n=10 #set sample size
x<-rnorm(n = n, mean = 100, sd = 15) #create sample from normal distribution


#plot data
ggplot(as.data.frame(x), aes(x))  + 
  geom_histogram(colour="black", fill="grey", aes(y=..density..), binwidth=2) +
  stat_function(fun=dnorm, args=c(mean=100,sd=15), size=1, color="red", lty=2) +
  #  geom_density(fill=NA, colour="black", size = 1) +
  xlab("IQ") + ylab("number of people")  + ggtitle("Data") + theme_bw(base_size=20) + 
  theme(panel.grid.major.x = element_blank(), axis.text.y = element_blank(), panel.grid.minor.x = element_blank()) + 
  geom_vline(xintercept=mean(x), colour="gray20", linetype="dashed") + 
  coord_cartesian(xlim=c(50,150)) + scale_x_continuous(breaks=c(50,60,70,80,90,100,110,120,130,140,150)) +
  annotate("text", x = mean(x), y = 0.02, label = paste("Mean = ",round(mean(x)),"\n","SD = ",round(sd(x)),sep=""), size=8)

#© Daniel Lakens, 2016. 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/