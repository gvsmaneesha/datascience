rm(list = ls())

getwd()

install.packages("DBI")

install.packages(DBI)

##Connect database
library(RMySQL)

#Connect to local SQL db from local machine
channel = dbConnect(MySQL(), user='root', password='password', dbname='edwisor')

#Return list of tables avilable in database
dbListTables(channel)

#extract data from database
RunEngine = dbSendQuery(channel, "select * from superstoreus_2015")

USstore_Data = fetch(RunEngine, n = -1)

#close pending query
dbHasCompleted(RunEngine)
dbClearResult(RunEngine)

#Extract data based on condition 
RunEngine = dbSendQuery(channel, "SELECT * FROM superstoreus_2015 WHERE Sales >= 1000")
USstore_Data = fetch(RunEngine, n = -1)

#close pending query
dbHasCompleted(RunEngine)
dbClearResult(RunEngine)

#Close Connection
dbDisconnect(channel)


##Let's connect to twitter
library(twitteR)
library(RCurl)
library(ROAuth)

#Connect R to twitter. Login to twitter account and creat app to get key and tokens
#https://apps.twitter.com/
api_key = "dD4i6dIp7mQLLALBd26rHw"
api_secret = "kSdjITYWdMGkK0RkfNtMNejuiOaOrqZtezOzFJ2NzQ"
access_token = "1945980420-WLnsskYxAdHM411Fnqry7Oubz0NVT2juwP6pySo"
access_token_secret = "Xwg1KbAKQzqeWvIXcxAE2FE2xMJpxhgdEuRH0N6MCJBN9"

setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret) #Press 0

#establish Connection
tweets = searchTwitter("education", since = "2016-06-01", until = "2016-07-31", n = 500)
tweets = searchTwitter('education', n = 1000, retryOnRateLimit = 1)

#Convert tweets list into data frame
tweets = twListToDF(tweets)

#Wrtite back tweets
write.csv(tweets, "tweets_Mar17-Mar30.csv", row.names = F)

#connect to facebook
 x = c("httr","rjson","httpuv")

install.packages(x)
library(Rfacebook)

#Generic functions 
#sort function
sort_fn = function(df, var){
          df = df[order(-df[var]),]
      return(df)
} 

Sort_data = sort_fn(tweets, "retweetcount")
Sort_data = sort_fn(USstore_Data, "Profit")

