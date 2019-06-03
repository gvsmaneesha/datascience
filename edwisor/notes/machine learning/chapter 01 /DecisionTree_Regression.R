
rm(list=ls(all=T))
setwd("Documents/datasets/")

#Load Libraries
library(rpart)
library(MASS)

#Load practice data
df = birthwt

write.csv(df,"df.csv",row.names = T)

#Divide the data into train and test
#set.seed(123)
train_index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[train_index,]
test =  df[-train_index,]
 
# ##rpart for regression
fit = rpart(bwt ~ ., data = train, method = "anova")

#Predict for new test cases
predictions_DT = predict(fit, test[,-10])

#MAPE
#calculate MAPE

MAPE = function(y, yhat){
             mean(abs((y - yhat)/y))
}

MAPE(test[,10], predictions_DT)

#Error Rate: 10.33
#Accuracy: 89.67

#variance Influencial Factor * VIF)
#formula 1/1-r2)

install.packages('usdm')
library(usdm)
vif(df[,-10])

#variance influential factor with corellation
vifcor(df[,-10],th = 0.9)

 #linear regression model 

lm_model = lm(bwt~.,data = train)

#summary of model 

summary(lm_model)
  
#prection model 

predcition_Lr = predict(lm_model , test[,1:9])

MAPE(test[,10],predcition_Lr)


