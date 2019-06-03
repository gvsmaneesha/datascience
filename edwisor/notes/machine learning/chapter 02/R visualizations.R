
##################################Feature Scaling################################################
#Normality check
#data looks normalized
hist(data$cnt)


# get the range for the x and y axis 
x <- range(data$mnth) 
y1 <- range(data$cnt) 

plot(c(data$cnt,data$temp),type = "o", col = "red", xlab = "Month", ylab = "Rain fall",
     main = "Rain fall chart")

library(sqldf)
install.packages("sqldf")
library("ggridges")
library('ggplot2')


season_summary_by_weekday <- sqldf('select season, weekday,avg(cnt) as count from data group by season, weekday')

#provides the aveage no.of users  by weak and by summer 
# on obsrving the following graph we can say the the users in the fall season are more compared to others
ggplot(season_summary_by_weekday,aes(x=weekday, y=count, color=season))+geom_point(data = season_summary_by_weekday, aes(group = count))+geom_line(data = season_summary_by_weekday, aes(group = season))+ggtitle("Bikes Rent By Season")

season_summary_by_yr<- sqldf('select season, yr ,avg(cnt) as count from data group by yr, season')


ggplot(season_summary_by_yr,aes(x = yr, y = count, fill = season, label = yr )) +
  geom_bar(stat = "identity") +
  #geom_text(size = 3, position = position_stack(vjust = 0.5))
  #geom_text(label=count,size = 3, position = position_stack(vjust = 0.5))
  
  
  ggplot(data,aes(x=windspeed,y=cnt))+geom_line(color = "red",size = 0.5)


ggplot(data=data, aes(x=yr, y=cnt)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=cnt), size=3.5)+
  theme_minimal()



ggplot(
  data, 
  aes(x = data$cnt, y = data$yr )
) +
  geom_density_ridges_gradient(
    aes(fill = ..x..), scale = 3, size = 0.3
  ) +
  scale_fill_gradientn(
    colours = c("#0D0887FF", "#CC4678FF", "#F0F921FF"),
    name = "bike rent"
  )+
  labs(title = 'monthly statistics') 
