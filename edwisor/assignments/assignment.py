#import the pandas library
import pandas as pd


#reading the CSV file and encoding it and the 2nd row values will be skipped
imdb_data = pd.read_csv('IMDB_data.csv', encoding = "ISO-8859-1",skiprows = [1]) 

#print the datatype details of the dataset
print(imdb_data.info())

#calculating the count and then renaming the columns and resetting the index values.to_frame is used to create dataframe and then reset_index() to create index values
df=imdb_data.Genre.value_counts().to_frame().reset_index().rename(columns={"Genre" : "Count", "index" : "Genre"})



#sort the genre by its name
df.Genre.sort_values(ascending=True)

#sort the genre by its name
imdb_data.Genre.sort_values(ascending=True)

    
imdb_data.imdbRating.convert_objects(convert_numeric=True)
imdb_data.imdbRating = imdb_data.imdbRating.convert_objects(convert_numeric=True)

imdb_data.imdbVotes = imdb_data.imdbVotes.convert_objects(convert_numeric=True)


imdb_data.imdbRating.dtype
imdb_data.imdbVotes.dtype


imdb_data['Variance'] = '0'

#Create new variable whose values should be square of difference between imdbrating and imdbvotes

#for i in range(len(imdb_data)):
    #imdb_data['Variance'].loc[i] = imdb_data['imdbVotes'].loc[i] - imdb_data['imdbRating'].loc[i]
    
print(imdb_data)

