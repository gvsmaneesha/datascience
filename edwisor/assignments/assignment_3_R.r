#remove all the objects stored
rm(list=ls())
#install.packages('plyr')
library(plyr)
#set current working directory
setwd("Documents")

#get working directory
df=(read.csv("IMDB_data.csv",header=T,sep=",",fileEncoding = "ISO-8859-1")[-2,])

#calculating the count of variables
Genre_count = count(df$Genre)


#create the column index
Genre_count = cbind(rownames(Genre_count),Genre_count)

#renaming the column names
colnames(Genre_count)<- c("sno","Genre","Count")

#sorting dataset by genre variable
df <- df[order(df$Genre),] 

#sorting dataset by genre variable

Genre_count <- Genre_count[order(Genre_count$Genre),] 

#changing the variable type to numeric
df$imdbRating = as.numeric(as.character(df$imdbRating))

#changing the variable type to numeric
df$imdbVotes = as.numeric(as.character(df$imdbVotes))

#Create new variable whose values should be square of difference between imdbrating and imdbvotes.

df$variance = ((df$imdbRating) - (df$imdbVotes))**2 




                    