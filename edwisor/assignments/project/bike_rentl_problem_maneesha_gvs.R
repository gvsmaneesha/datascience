rm(list=ls(all=T))
setwd("Documents/datasets/")

######################## Read the data #######################################################################
df = read.csv("days.csv", header = T, na.strings = c(" ", "", "NA"))
data = df
str(df)
head(df,5)
########################################### Data Preprocessing ##############################################

#remove registered and casual users since the final count is available in the "cnt" variable
#instant variable should be removed as it is just a serial number and has no value in the analysis.
df = df[,-c(1)]

#converting numeric variables to factors
names <- c("season","yr","mnth","holiday","weekday","workingday","weathersit")
df[,names] <- lapply(df[,names] , factor)

data = df
###################convert numberical varaibles to categorical variables###############################
levels(data$season)<- list( "spring"="1", "Summer"="2","fall" ="3","winter" ="4")
levels(data$yr)<-list("2011" = "0" ,"2012" = "1")
levels(data$mnth)<-list("JAN" = "1" ,"FEB" = "2" ,"MAR" = "3" , "APR" ="4" ,"MAY" = "5" , "JUNE" = "6" , "JUL" = "7","AUG" = "8","SEP" = "9","OCT" = "10","NOV" = "11","DEC" = "12")
levels(data$holiday)<-list("yes"= "0","no"="1")
levels(data$weekday)<-list("sun" = "0","mon" = "1","tue" = "2","wed" = "3","thur" = "4","fri" = "5","sat" = "6")
levels(data$workingday)<-list("no" = "0" , "yes" = "1")
levels(data$weathersit)<-list("normal" = "1","moderate" = "2","heavy" = "3","extreme" = "4")

##################################Missing Values Analysis###############################################
sum(is.na(df))

#hence no missing values , hence there is no need of missing value analysis

##################################Feature Selection#####################################################
## Correlation Plot 

numeric_index = sapply(df,is.numeric) #selecting only numeric

library('corrgram')#load the library to plot correlation graph
corrgram(df[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


#from correlation graph it clear the the temp and atemp are highly correlated with each other 
#hence we can consider temp alone and remove atemp

df = df[,-c(10)]

##################################Feature Scaling######################################################
#Normality check
#data looks normalized
hist(df$cnt)
###########################Outlier Analysis############################################################
numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df[,numeric_index]#select only numerical data

cnames = colnames(numeric_data)#store the column names of numeric data

#there are 6 numerical variables 

library(ggplot2)
 for (i in 1:length(cnames))
{
   assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(df))+ 
            stat_boxplot(geom = "errorbar", width = 0.5) +
            geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                         outlier.size=2, notch=FALSE) +
            theme(legend.position="bottom")+
            labs(y=cnames[i],x="cnt")+
            ggtitle(paste("Box plot for",cnames[i])))
 }
 ## Plotting plots together
 gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,ncol=5)

 
 ##from the outlier graph it is clear that the outliers are present for windspeed = 2, hum = 13, casual = 44 variables
#Replace all outliers with NA and impute
for(i in cnames){
   val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
   print(length(val))
    df[,i][df[,i] %in% val] = NA
  }
#KNN imputation is used to replace the outliers
df = knnImputation(df, k = 3)

##after KNN imputation , there are not outliers present

######################################### analysis via visualizations#######################################
library('sqldf') # sql db is used to group the values and do further analysis
library('ggplot2')# used for visualizations 

season_summary_by_weekday <- sqldf('select season, weekday,avg(cnt) as count from data group by season, weekday')


#provides the average no.of users  by weak and by summer
# on observing the following graph we can say the the users in the fall season are more compared to others
ggplot(season_summary_by_weekday,aes(x=weekday, y=count, color=season))+geom_point(data = season_summary_by_weekday, aes(group = season))+geom_line(data = season_summary_by_weekday, aes(group = season))+ggtitle("Bikes Rent By Season")

count_summary_by_temp<- sqldf('select weathersit,temp,avg(cnt) as count from data group by temp,weathersit')

#bikes count is more when the temperature is in the range (0.25 to 0.75) and also when weathersit are moderate, the maximum bikes counts
#is observed when temp is in range of 0.5 and of moderate weathersit
#
ggplot(count_summary_by_temp,aes(x = temp, y = count, fill = weathersit, label = weathersit)) +
  geom_bar(stat = "identity",width = 0.003)+ggtitle("Bikes Rental count By temp")

count_summary_by_hum<- sqldf('select weathersit,hum,avg(cnt) as count from data group by hum,weathersit')

ggplot(count_summary_by_hum,aes(x = hum, y = count, fill = weathersit, label = weathersit)) +
  geom_bar(stat = "identity",width = 0.003)+ggtitle("Bikes Rental count By humidity")


count_summary_by_windspeed<- sqldf('select weathersit,windspeed,avg(cnt) as count from data group by windspeed,weathersit')

ggplot(count_summary_by_windspeed,aes(x = windspeed, y = count, fill = weathersit, label = weathersit)) +
  geom_bar(stat = "identity",width = 0.003)+ggtitle("Bikes Rental count By windspeed")

count_summary_by_workingdays<- sqldf('select workingday,avg(cnt) as count from data group by workingday')

###################################Model Development###########################################################
train_index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[train_index,]
test =  df[-train_index,]

#Load Libraries
library(rpart)
library(MASS)
# ##rpart for regression
fit = rpart(cnt~ ., data = train, method = "anova")

#Predict for new  test cases
predictions_DT = predict(fit, test[,-14])

summary(predictions_DT)

#calculate MAPE

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}
MAPE(test[,14], predictions_DT)

#error rate : 12.56
#accuracy: 87.44


write.csv(file="test_sample.csv",test)
write.csv(file="bikerent_sample.csv",predictions_DT)
#######end########################################################################################################



