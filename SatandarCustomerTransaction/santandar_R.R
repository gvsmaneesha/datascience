rm(list = ls())
getwd()
setwd("~/Documents/datasets")
################################### load library files ################################################################
x = c("geosphere","stringr","DMwR","caret","rpart","MASS","usdm",'randomForest','sqldf','ggplot2',"xgboost",'Matrix','pie3D',"C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)
###################################load library files ################################################################
train1 = read.csv("Train_dataset.csv", header = TRUE)
test1 = read.csv("Test_dataset.csv", header = TRUE)
train = train1[,-c(1)]
test = test1[,-c(1)]
target = train$target
test_ID = test$ID_code

(t <- table(train$target) / nrow(train))
require(plotrix)
l <- paste(c('Happy customers\n','Unhappy customers\n'), paste(round(t*100,2), '%', sep=''))
pie3D(t, labels=l, col=c('green','red'), main='Santander customer satisfaction dataset', theta=1, labelcex=0.8)


############################## DATA PREPROCESSING ################################################################################
# remove constant features
for (i in names(train)) {
  if (length(unique(train[[i]])) == 1) {
    cat(i, "is constant in train (", unique(train[[i]]),"). We delete it.\n")
    train[[i]] <- NULL
    test[[i]] <- NULL
  }
}

# remove identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equal.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train[,-c(1)]), toRemove)

train <- train[,feature.names]
test <- test[,feature.names]

# Removing highly correlated variables
cor_v <- abs(cor(train))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_f <- as.data.frame(which(cor_v > 0.85, arr.ind = TRUE))
#train <- train[,-unique(cor_f$row)]
#test <- test[,-unique(cor_f$row)]

image(cor_v)

anyNA(train)
anyNA(test)
summary(train)
#############################  MODEL DEVELOPMENT ############################################################################
train$target = target
train$target = as.factor(train$target)
#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(train$target, p = .80, list = FALSE)
Train2 = train[ train.index,]
Test2 = train[-train.index,]

##Decision tree for classification
#Develop Model on training data
#C50_model = C5.0(target ~., Train2, trials = 10, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
#write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
#C50_Predictions = predict(C50_model,subset(mydata, select = -c(target)), type = "class")

##Evaluate the performance of classification model
#ConfMatrix_C50 = table(test$target, C50_Predictions)
#confusionMatrix(ConfMatrix_C50)

#Accuracy: 90.89%
#FNR: 63.09%

#################################################### END OF DECISION TREES #################################################

###Random Forest
#RF_model = randomForest(target ~ ., Train2, importance = TRUE, ntree = 2)

#Presdict test data using random forest model
#RF_Predictions = predict(RF_model,subset(Train2, select = -c(target)))

##Evaluate the performance of classification model
#ConfMatrix_RF = table(Test2$responded, RF_Predictions)

#print(confusionMatrix(ConfMatrix_RF))

#Accuracy: 93.89%
#FNR: 58.09%

############################################ END OF RANDOM FOREST ########################################################
library('usdm')

#Logistic Regression
logit_model = glm(target ~ ., data = Train2, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = Test2, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_log = table(Test2$target, logit_Predictions)
print(confusionMatrix(ConfMatrix_log))

#accuracy - 91.4
#FPR - 32.2

#################################### END OF LOGISTIC REGRESSION ############################################################
#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(target ~ ., data = Train2)

#predict on test cases #raw
NB_Predictions = predict(NB_model,subset(Test2, select = -c(target)), type = 'class')


#Look at confusion matrix
Conf_matrix = table(Test2$target,predicted = NB_Predictions)
print(confusionMatrix(Conf_matrix))

#Accuracy: 92.16
#FPR: 29.57
#################################################### END OF NAIVE BAYES #####################################################

logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
test$target = ifelse(logit_Predictions > 0.5, 1, 0)

write.csv(test,"submission_R.csv")
