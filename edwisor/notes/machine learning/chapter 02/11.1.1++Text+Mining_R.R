#clear the environment
rm(list=ls())

#check for current working directory
getwd()

#Set working directory
setwd("Documents/datasets/")

#Load days.csv data set
days = read.csv("days.csv", header = T)

sum((days.is.na()))
#Load defined stop words
#stop_words = read.csv("stopwords.csv", header = T)
#names(stop_words) = "StopWords"

#Delete the leading spaces with help of str_trim
post$Post = str_trim(post$Post)

#Select only text column
post = data.frame(post[1:200,2])
names(post) = "comments"
post$comments = as.character(post$comments)

##Pre-processing
#convert comments into corpus present in the tm library   
postCorpus = VCorpus(VectorSource(post$comments))

#The code should still be working. You get a warning, not an error. This warning only appears when you have a corpus based on a VectorSource in combination when you use Corpus instead of VCorpus.

#The reason is that there is a check in the underlying code to see if the number of names of the corpus content matches the length of the corpus content. With reading the text as a vector there are no document names and this warning pops up. And this is only a warning, no documents have been dropped.

writeLines(as.character(postCorpus[[1]]))
writeLines(as.character(postCorpus[[3]]))
postCorpus = tm_map(postCorpus, PlainTextDocument)

#case folding
postCorpus = tm_map(postCorpus, tolower)




#remove stop words
postCorpus = tm_map(postCorpus, removeWords, stopwords('english'))


#remove punctuation marks
postCorpus = tm_map(postCorpus, removePunctuation)


#me is removed
writeLines(as.character(postCorpus[[2]]))


#remove numbers
postCorpus = tm_map(postCorpus, removeNumbers)

typeof(postCorpus)
#remove unnecesary spaces
postCorpus = tm_map(postCorpus, stripWhitespace)

# #convert into plain text
postCorpus = tm_map(postCorpus, PlainTextDocument)
# 
# #create corpus
postCorpus = VCorpus(VectorSource(postCorpus))

#Build document term matrix
tdm = TermDocumentMatrix(postCorpus)
#tdm_min = TermDocumentMatrix(postCorpus, control=list(weighting=weightTfIdf, minWordLength=4, minDocFreq=10))

#Convert term document matrix into dataframe
TDM_data = as.data.frame(t(as.matrix(tdm)))
# why transpose is done,since columns are considered as documents we to make it as rows

class(tdm)

str(TDM_data)

##calculate the terms frequency
words_freq = rollup(tdm, 2, na.rm=TRUE, FUN = sum)

#Convert into matrix
words_freq = as.matrix(words_freq)

#Convert to proper dataframe
words_freq = data.frame(words_freq)

#Convert row.names into index
words_freq$words = row.names(words_freq)

#row names
row.names(words_freq) = NULL
words_freq = words_freq[,c(2,1)]

#column names
names(words_freq) = c("Words", "Frequency")

#Most frequent terms which appears in atleast 700 times
findFreqTerms(tdm, 5)

##wordcloud 
#copy of corpus
postCorpus_WC = postCorpus

pal2 = brewer.pal(8,"Dark2")

png("wordcloud_v3.png", width = 12, height = 8, units = 'in', res = 300)
wordcloud(postCorpus_WC, scale = c(5,.2), min.freq = 3 , max.words = 150, random.order = FALSE, rot.per = .15, colors = pal2)
dev.off()

#Remove the defined stop words
postCorpus_WC = tm_map(postCorpus_WC, removeWords, c('will', 'also', 'can',
                                               stopwords('english')))

stop_words= data.frame(StopWords =c('will', 'also', 'can',
                                    stopwords('english')))

postCorpus_WC = tm_map(postCorpus_WC, removeWords, stop_words$StopWords)


#sentiment Analysis
#Another method
library(RSentiment)
post = data.frame(post[1:200,])
names(post) = 'comments'

df = calculate_sentiment(post$comments)

#Another method
#Install sentiment library using binay source
install.packages("Rstem_0.4-1.tar.gz", repos = NULL, type="source")
install.packages("sentiment_0.2.tar.gz", repos = NULL, type="source")

library('Rstem')
library(sentiment)

#classifying the corpus as negative and positive and neutral
polarity = classify_polarity(post$comments, algorithm = "bayes", verbose = TRUE)
polarity = data.frame(polarity)

 #Attached sentiments to the comments
newdocs = cbind(post, polarity)

#Pie chart to visualise polarity
df = data.frame(table(newdocs$BEST_FIT))

#Interactive visualisations using plotly
library(plotly)

plot_ly(df, labels = ~Var1, values = ~Freq, type = 'pie') %>%
        layout(title = 'Sentiment Analysis',
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


a<- "I love to play football bcz my favourite sportsperson is ronaldo bcz i love cars"
b<- gsub("bcz","because",a)
b


