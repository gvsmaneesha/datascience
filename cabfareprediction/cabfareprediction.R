rm(list = ls())
getwd()
setwd("~/Documents/datascience/cabfareprediction")
###################################load library files ################################################################
x = c("geosphere","stringr","DMwR","caret","rpart","MASS","usdm",'randomForest','sqldf','ggplot2')
install.packages(x,repos='http://cran.us.r-project.org')
lapply(x, require, character.only = TRUE)
rm(x)
###############################read train data########################################################################
train=read.csv("train_cab.csv",header = T,na.strings = c(""," ",NA))
test=read.csv("test.csv",header = T,na.strings = c(""," ",NA))
#####################data preprocessing and EDA methods ##############################################################
cat("no. of missing values in train dataset")
colSums(is.na(train))

cat("no. of missing values in test dataset")
colSums(is.na(test))

#################################### type conversions ###############################################################
cnames = c("pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude")

  
for(i in 1:ncol(train)){
  
  if(class(train[,i]) == 'list'){
    
    train[,i] = as.numeric(train[,i])
    
  }
}
  train$pickup_datetime = as.character(train$pickup_datetime)
  train$passenger_count = as.integer(as.character(train$passenger_count))

  
  for(i in 1:ncol(test)){
    
    if(class(train[,i]) == 'list'){
      
      test[,i] = as.numeric(test[,i])
      
    }
  }
  test$pickup_datetime = as.character(test$pickup_datetime)
  test$passenger_count = as.integer(as.character(test$passenger_count))
 
#######################################################################################################################
cord = c("pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude")
print("set the range of latitude and longitude for the newyork country")
lat_min=37
lat_max=45.0153
lon_min=-79.7624
lon_max=-71.7517
radius = 6371
cat("latitude limits",lat_min,",",lat_max)
cat("longitude limits",lon_min,",",lon_max)

data_cleaning_stage_cooridnates<-function(df){
  
  outliers<-function(x,l,r){if(x<l | x > r){return(NA)}else{return(x)}}
  
  df$pickup_latitude= lapply(df$pickup_latitude,function(x){ outliers(x,lat_min,lat_max)})
  df$dropoff_latitude= lapply(df$dropoff_latitude,function(x){ outliers(x,lat_min,lat_max)})
  df$pickup_longitude= lapply(df$pickup_longitude,function(x){ outliers(x,lon_min,lon_max)})
  df$dropoff_longitude= lapply(df$dropoff_longitude,function(x){ outliers(x,lon_min,lon_max)})

  df=na.omit(df)
  
  for(i in 1:ncol(df)){
    
    if(class(df[,i]) == 'list'){
      
      df[,i] = as.numeric(df[,i])
      
    }
  }
  
  
  radians=function(x){
    x = as.numeric(x)
  return(x*pi/180)}
  
  distance=function(lat1,lat2,long1,long2)
  {
    dlat = abs(radians(lat1)-radians(lat2))
    dlong = abs(radians(long1)-radians(long2))
    t1 = (sin(dlat/2)**2)+(cos(radians(lat1))*cos(radians(lat2))*sin(dlong/2)**2)
    t2 = 2*(atan2(sqrt(t1),sqrt(1-t1)))
    return(abs(6371*t2))
    
  }

  for(i in 1:nrow(df)){df$distance[i] = distance(df$pickup_latitude[i],df$dropoff_latitude[i],df$pickup_longitude[i],df$dropoff_longitude[i])}
  print("end of co-ordinates preprocessing")
  return(df)
}

data_cleaning_stage_date_time<-function(df){
      for(i in 1:nrow(df)){
        
        x=as.character(df$pickup_datetime[i])
        d = str_detect("2015-01-27 13:08:24 UTC", "[1-2][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9] [0-2][0-3]:[0-5][0-9]:[0-5][0-9] UTC")
        if(length(x)==length("2009-06-15 17:26:21 UTC")){
            if(d){
            df$year[i] = as.integer(substring(strsplit(x," "), 4,7))
            df$month[i]=as.integer(substring(strsplit(x," "), 9,10))
            df$day[i]=as.integer(substring(strsplit(x," "), 12,13))
            }
          else{return(NA)}
        }else{return(NA)}
        
      }
  df = subset(df,select = -c(pickup_datetime))
  return(df)
}
################################################end of data Preprocessing methods #####################################
train = data_cleaning_stage_cooridnates(train)
train = data_cleaning_stage_date_time(train)
train$fare_amount  = as.numeric(as.character(train$fare_amount))
train = train[train$fare_amount > 2.5 & train$fare_amount < 250,]
train = train[train$passenger_count > 0 & train$passenger_count < 7 ,]
test = data_cleaning_stage_cooridnates(test)
test = data_cleaning_stage_date_time(test)
test = test[complete.cases(test[ ,"passenger_count"]),]
test = test[test$passenger_count > 0 & test$passenger_count < 7 ,]
train=na.omit(train)
sum(is.na(train))
sort(train$fare_amount,decreasing = TRUE)
############################################### end of data cleaning ################################################

fareamount_by_year <- sqldf('select year, passenger_count , avg(fare_amount) as fare from train group by year,passenger_count')
ggplot(fareamount_by_year,aes(x=year, y=fare, color=passenger_count))+geom_point(data = fareamount_by_year, aes(group = passenger_count))+geom_line(data = fareamount_by_year, aes(group = passenger_count))+ggtitle("cab fare by year , passenger_count")

fareamount_by_month <- sqldf('select month, avg(fare_amount) as fare from train group by month')

fareamount_by_distance_year <- sqldf('select year, avg(fare_amount) as fare from train group by year, fare_amount')
ggplot(fareamount_by_distance_year,aes(x = year, y = fare, fill = year, label = year )) +
  geom_bar(stat = "identity",width = 0.20)+ggtitle(" avg cabfare amount vs avg distance by year")
################################################## end of data visualizations ######################################
MAPE = function(actual, prediction){return(mean(abs((actual - prediction)/actual))*100)}
RMSE=function(actual, predicted){ return(sqrt(mean((actual - predicted)**2))) }
MSE=function(actual, predicted){  return(mean((actual - predicted)**2)) }

error_metrics <- data.frame(matrix(ncol = 3, nrow = 3))
x <- c("DecisionTree", "RandomForest", "LinearRegression")
colnames(error_metrics) <- x
x <- c("MAPE", "MSE", "RMSE")
rownames(error_metrics)<-x
############################################ end of error metrics functions #######################################
train.index = createDataPartition(train$fare_amount, p = .20, list = FALSE)
train1 = train[ train.index,]
test1  = train[-train.index,]
rm(train.index)
fit = rpart(fare_amount ~ ., data = train1, method = "anova")
predictions_DT = round(as.numeric(predict(fit,test1[,-1])),2)
MAPE(test1[,1],predictions_DT)
MSE(test1[,1],predictions_DT)
RMSE(test1[,1],predictions_DT)
########################################## end of decision tree  mode ###########################################
lm_model = lm(fare_amount~.,data = train1)
vif(train1[,-1])
vifcor(train1[,-1],th = 0.9)
summary(lm_model)
predcition_Lr = predict(lm_model, test1[,2:10])
MAPE(test1[,1],predcition_Lr)
MSE(test1[,1],predcition_Lr)
RMSE(test1[,1],predcition_Lr)
############################################ end of linear regression #########################################
RF_model = randomForest(fare_amount ~ ., train1, importance = TRUE, ntree = 500)
RF_Predictions = predict(RF_model, test1[,-1])
round(MAPE(test1[,1],RF_Predictions),2)
round(MSE(test1[,1],RF_Predictions),2)
round(RMSE(test1[,1],RF_Predictions),2)
########################################## end of random forest ##############################################
test$fare_amount = round(predict(RF_model, test),2)

write.csv(test,"test_fare_amount_R.csv")

print("the test fare amount has been successfully calculated and exported")



