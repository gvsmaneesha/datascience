##Connect database

rm(list = ls())
library("mongolite")
con = mongo(collection = "test", db = "test", url = "mongodb://localhost")
con$insert(iris)

temp = data.frame(c("maneesha","gvs","age"))
con$insert(temp)
con$find()
