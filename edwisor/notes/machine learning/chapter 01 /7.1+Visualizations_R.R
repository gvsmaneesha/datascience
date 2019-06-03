rm(list=ls())

#set working directory
setwd("Documents/datasets/")

#load libraries
library("ggplot2")
library("scales")
library("psych")
library("gplots")

#Load input data
Marketing_data = read.csv("marketing_tr.csv", header=T)

#Univariate 
#Bar plot(categorical data)
#If you want count then stat="bin"
#aes_string function to plot x and y variables

# gemo_bar function inorder to set the type of chart and the customization purpose

#Xlab and ylab are the function to rename the x axis and y axis names

#ggtitle to rename the chart 

#scale_y_continuous -> it will indicates the splits 


ggplot(Marketing_data, aes_string(x = Marketing_data$profession)) +
  geom_bar(stat="count",fill =  "darkblue") + theme_bw() +
  xlab("Profession") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("Marketing Campaign Analysis") +  theme(text=element_text(size=10))


#Histogram
#geom_histogram --> fill (bacgrund color ) and color  line color of the histogram
#
ggplot(Marketing_data, aes_string(x = Marketing_data$custAge)) + 
  geom_histogram(fill="cornsilk", colour = "brown") + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("Age") + ylab("Frequency") + ggtitle("Marketing_data: Age") +
  theme(text=element_text(size=20))

#Box plot
ggplot(Marketing_data, aes_string(x = Marketing_data$responded, y = Marketing_data$custAge , fill = Marketing_data$responded)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 0.5) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("Responded") + ylab("Customer Age") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))

#Multivariate #Scatter Plot
ggplot(Marketing_data, aes_string(x = Marketing_data$campaign, y = Marketing_data$custAge)) + 
  geom_point(aes_string(colour = Marketing_data$responded, shape = Marketing_data$profession),size = 4) +
  theme_bw()+ ylab("Customer Age") + xlab("Campaign") + ggtitle("Scatter plot Analysis") + 
  theme(text=element_text(size=25)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10)) +
  scale_y_continuous(breaks=pretty_breaks(n=10)) +
  scale_colour_discrete(name="Responded")+
  scale_shape_discrete(name="Profession")

