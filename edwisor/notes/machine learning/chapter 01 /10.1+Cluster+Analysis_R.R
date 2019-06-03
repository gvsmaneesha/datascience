rm(list=ls())

#set current working directory
setwd("Documents/datasets/")

#load libraries
library(NbClust)

#Load data
df = iris

write.csv(df,'iris.csv')

#standadize the data
df_new = data.frame(scale(df[-5]))

head(df_new)
#extract number of clusters to bulid
NBclust_res = NbClust(df_new, min.nc=2, max.nc=15, method = "kmeans")
sum(NBclust_res$Best.n[1,] == 3)



#Barplot to analys the optimum clusters
barplot(table(NBclust_res$Best.n[1,]),
        xlab="#Clusters", ylab="#Criteria",
        main="#Clusters Chosen by 26 Criteria")

#K-mean clustering
kmeans_model = kmeans(df_new, 3, nstart=25)

#Summarize cluster output
kmeans_model

#How well your kmeans is??
Cluster_accuracy = table(df$Species, kmeans_model$cluster)

Cluster_accuracy









