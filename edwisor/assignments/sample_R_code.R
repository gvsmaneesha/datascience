app
print(" The datatypes present in the population data frame are ",str(df)) 
vec=c(1,2,3,4,4,4,4,4,4,41,1,1,1,1,3,3,3)
str(vec)
vec=as.factor(vec)
print("the data type of vector has been changed",str(vec))
vector=c(1,2,3,"level1")
lis=list(1,2,3,"level1")
str(lis)
str(vec)
str(df$Year)
df$ctgyr[df$Year > 2012 & df$Year < 2015]="olddata"
df$ctgyr[df$Year > 2015 & df$Year <= 2017]="newdata"
str(df$ctgyr)
df$ctgyr=as.factor(df$ctgyr)
str(df$ctgyr)
str(df)
df$Industry_aggregation_NZSIOC=as.character(df$Industry_aggregation_NZSIOC)
str(df$Industry_aggregation_NZSIOC)
voice=c(rep("medium",30),rep("poor",30),rep("Rich",40))
#levels1 = levels(factor(voice))
#levels1
#lenght1 = length(levels1)
#lenght1
#lables1=(1:lenght1)
#lables1
#voice_factor=factor(voice,lables1)
#voice_factor
#voice_numeric=as.numeric(voice_factor)
#voice_numeric */
voice_factor=factor(voice,labels = (1:length(levels(factor(voice)))))
df=df[order(df$value),]
rm(list = ls())



