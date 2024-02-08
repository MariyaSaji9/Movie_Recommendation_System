# **MOVIE** **RECOMMENDATION SYSTEM**



## IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount("/content/drive")

## LOAD THE DATASET

net_mov = pd.read_csv("/content/drive/MyDrive/movie_pro/netflix_movies.csv")
net_mov

## ANALYSING THE DATA

net_mov.shape

net_mov.info()

#out of 12 columns 11 are object type and 1 is numerical

# missing value analysis
miss = net_mov.isnull().sum()
miss

#director, cast country,date_added, rating, duration columns have missing values

per = (miss/len(net_mov))*100
per

#Handling missing values

net_mov['country'].value_counts()

net_mov['date_added'].value_counts()

#filling country and date_added with mode
net_mov['country'] = net_mov['country'].fillna(net_mov['country'].mode()[0])
net_mov['date_added'] = net_mov['date_added'].fillna(net_mov['date_added'].mode()[0])
net_mov['rating'] = net_mov['rating'].fillna(net_mov['country'].mode()[0])
net_mov['duration'] = net_mov['duration'].fillna(net_mov['duration'].mode()[0])

#we are dropping the rows having missing values in director and cast because it's not good to fill using mode and any other techniques.
net_mov = net_mov.dropna(how = 'any', subset = ['director', 'cast'])

net_mov.shape

net_mov.isnull().sum()

net_mov

#for interpreting the dataset easily some cleaning and preprocessing steps are performing

net_mov = net_mov.rename(columns={"listed_in":"Genre"})
net_mov['Genre'] = net_mov['Genre'].apply(lambda x: x.split(",")[0])

#renaming and splitting the values into two columns
net_mov['year_added'] = net_mov['date_added'].apply(lambda x: x.split(" ")[-1])

net_mov['month_added'] = net_mov['date_added'].apply(lambda x: x.split(" ")[0])
net_mov['month_added']

net_mov['country'] = net_mov['country'].apply(lambda x: x.split(",")[0])

net_mov

#date_added and show_id are not needed more
net_mov.drop(['date_added'], axis = 1, inplace = True)
net_mov.drop(['show_id'], axis = 1, inplace = True)

net_mov.shape

net_mov['type'].value_counts()

net_mov['rating'].value_counts()

# DATA ANALYSIS USING PLOTS

#movies vs tv show
sns.set(style="darkgrid")
sns.countplot(x="type", data= net_mov, palette="Set1")

netflix provides movies than tv show

# Rating distribution for Movies and TV Shows
plt.figure(figsize=(12, 6))
sns.countplot(x="rating", data=net_mov, hue="type", palette="viridis")
plt.title('Rating Distribution for Movies and TV Shows')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#In movie type high rating is for TV-MA, second rating goes to TV-14.
#In tv shows high rating goes to TV-MA, next is to TV-14

# Yearly analysis plot
plt.figure(figsize=(16, 8))
sns.countplot(x="year_added", data=net_mov, hue="type", palette="muted")
plt.title('Yearly Analysis of Movies and TV Shows Added on Netflix')
plt.xlabel('Year Added')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#2019, 2020, 2018, 2021 are the years netflix added large amount of movies.
#2020,2021, 2019, 2017 are the years netflix added large amount of tv show

# Movie and TV Show Duration Analysis
plt.figure(figsize=(16, 10))
sns.histplot(x="duration", data=net_mov, hue="type", multiple="stack", palette="pastel", binwidth=5)
plt.title('Movie and TV Show Duration Analysis on Netflix')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Convert 'duration' column to numeric
net_mov['duration'] = pd.to_numeric(net_mov['duration'].str.extract('(\d+)')[0])

# KDE Plot for Movie Durations
plt.figure(figsize=(12, 8))
sns.kdeplot(x="duration", data=net_mov[net_mov['type'] == 'Movie'], fill=True, common_norm=False, palette="Set2", label='Movies')
plt.title('KDE Plot of Movie Durations on Netflix')
plt.xlabel('Duration (minutes)')
plt.ylabel('Density')
plt.legend()
plt.show()

#the majority of movies are having 75 to 120 as the duration of the movie

#Making two new dataframes for movies and TV shows


movie_df = net_mov[net_mov['type'] == 'Movie']
tv_df = net_mov[net_mov['type'] == 'TV Show']

movie_df

tv_df

#movie duration over years
duration_year = movie_df.groupby(['release_year']).mean()
duration_year = duration_year.sort_index()

plt.figure(figsize=(15,6))
sns.lineplot(x=duration_year.index, y=duration_year.duration.values)
plt.box(on=None)
plt.ylabel('Movie duration in minutes');
plt.xlabel('Year of released');
plt.title("Movie Duration over years", fontsize=20, color='red');

#from 1960 to 1965 the Movie durations were 200 minutes.
#after 1965 the durations became shorter.
#from 1980 movie durations is between 100-150 minutes.

# Countplot for Number of Seasons in TV Shows
plt.figure(figsize=(14, 8))
sns.countplot(x="duration", data=tv_df, palette="deep")
plt.title('Number of Seasons in TV Shows on Netflix')
plt.xlabel('Number of Seasons')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#135 TV shows have only 1 season.
#around 15 above tv shows have 2 seasons.

#Extract the columns title and duration from tv_df
columns=['title','duration']
tv_shows = tv_df[columns]

#sort the dataframe by number of seasons
tv_shows = tv_shows.sort_values(by='duration',ascending=False)
tv_shows
top20 = tv_shows[0:20]
top20

plt.figure(figsize=(10,6))
top20.plot(kind='bar',x='title',y='duration', color='orange')

#Supernatural , Naruto,The Great British Baking, and Call the Midwife has the highest numbers of seasons.

#countries based on content creation of movies
plt.figure(figsize=(18,8))
sns.countplot(x="country", data=movie_df, palette="hls", order=movie_df['country'].value_counts().index[0:15])

#United States , India , United Kingdom are in the first positions in creating movies.


#countries based on content creation for tv shows
plt.figure(figsize=(22,8))
sns.countplot(x="country", data=tv_df, palette="hls", order=tv_df['country'].value_counts().index[0:15])

  #United States,United Kingdom,South Korea creates most of the TV Shows.

# Genres of contents created by Countries
columns=['Genre','country']
country_genre = net_mov[columns]

country_genre

country_genre['Genre'].value_counts()

#United states produces most amount of content in 'Comedies' and 'Childern & Family movies' Genres.

#number of content released in each year
release = net_mov['release_year'].value_counts()
release = release.sort_index(ascending=True)
plt.figure(figsize=(9,7))
plt.plot(release[-11:-1])
plt.scatter(release[-11:-1].index, release[-11:-1].values, s=0.5*release[-11:-1].values, c='Red')
plt.box(on=None)
plt.xticks(rotation = 60)
plt.xticks(release[-11:-1].index)
plt.title('Number of Content Released by Year', color='green', fontsize=10)

2017 and 2018 released around 700 contents

#Directors with most number of Movies produced
plt.figure(figsize=(10,8))
sns.barplot(y= movie_df.director.value_counts()[:10].sort_values().index, x=movie_df.director.value_counts()[:10].sort_values().values)
plt.title('Director with most number of movies', color='red', fontsize=18)
plt.xticks(movie_df.director.value_counts()[:10].sort_values().values)
plt.xlabel('Number of Movies Released')

#Director Raul Campos,Jan Suter directed highest number of movies: 18

#directors and tv shows
plt.figure(figsize=(10,8))
sns.barplot(y= tv_df.director.value_counts()[:10].sort_values().index, x=tv_df.director.value_counts()[:10].sort_values().values)
plt.title('Director with most number of TV Shows', color='green', fontsize=10)
plt.xticks(tv_df.director.value_counts()[:10].sort_values().values)
plt.xlabel('Number of Shows Released')

#Alastair Fothergill released highest number of TV shows:3

#poplular genre
plt.figure(figsize=(24,16))
sns.barplot(x= net_mov.Genre.value_counts()[:10].sort_values().index, y=net_mov.Genre.value_counts()[:10].sort_values().values,palette='hls')
plt.title('Most Popular Genre', color='Blue', fontsize=10)
plt.yticks(net_mov.Genre.value_counts()[:10].sort_values().values)
plt.xlabel('GENRES')
plt.ylabel('Number of contents')

#Dramas are most poplular.
#Comedies are second most popular

#popular actors
plt.figure(figsize=(18,14))
sns.barplot(y= net_mov.cast.value_counts()[:15].sort_values().index, x=net_mov.cast.value_counts()[:15].sort_values().values,palette='gnuplot_r')
plt.title('Top Actor/Actresses on Netflix', color='purple', fontsize=30)
plt.xticks(net_mov.cast.value_counts()[:10].sort_values().values)
plt.ylabel('Actors', fontsize=10)
plt.xlabel('Content counts', fontsize=10)

#Vatsal Dubey, Julie Tejwani, Rupa Bhimani, Jigna Bhardwaj, Rajesh Kava, Mousam, Swapril has highest number of movies and Tv shows.

#content based on months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]
df_copy = net_mov.groupby('year_added')['month_added'].value_counts().unstack().fillna(0)[month_order].T


plt.figure(figsize=(10, 7), dpi=200)
plt.pcolor(df_copy, cmap='BrBG', edgecolors='white', linewidths=2) # heatmap
plt.xticks(np.arange(0.5, len(df_copy.columns), 1), df_copy.columns, fontsize=7, fontfamily='serif')
plt.yticks(np.arange(0.5, len(df_copy.index), 1), df_copy.index, fontsize=7, fontfamily='serif')

plt.title('Netflix Contents Update', fontsize=12, fontweight='bold')
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")

cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_on()
plt.show()

#more contents released on November, July, December and January.

# RECOMMENDATION SYSTEM

## Content Based Filtering

### Description based Recommender

net_mov['description']

#We can calculate similarity scores for all movies based on their descriptions.

#We convert descriptions into TF-IDF vectors to gauge word importance, reducing the impact of frequently occurring words in summaries for more accurate similarity scores.

from sklearn.feature_extraction.text import TfidfVectorizer
#Define a TF-IDF Vectorizer Object. Remove all stopwords
tfidf = TfidfVectorizer(stop_words='english')

net_mov['description'].isnull().sum()

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(net_mov['description'])
#Output the shape of tfidf_matrix
tfidf_matrix.shape

#Using the TF-IDF vectorizer, we directly compute cosine similarity scores by leveraging sklearn's linear_kernel(), which is faster than cosine_similarities().

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(net_mov.index, index=net_mov['title']).drop_duplicates()

#function for getting the 10 most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return net_mov['title'].iloc[movie_indices]

get_recommendations('My Little Pony: A New Generation')

get_recommendations('Supernatural')

#We have used one feature only. we can see these are not so accurate, so we can try to add more metrics to improve model performance.

## title, description, Genre, cast, director, based Recommender System

#From the Genre,cast and director features, we need to extract the three most important actors, the director and genres associated with that movie.

features=['Genre','director','cast','description','title']
filters = net_mov[features]

#Cleaning the data by making all the words in lower case.
def clean_data(x):
        return str.lower(x.replace(",", " "))

for feature in features:
    filters[feature] = filters[feature].apply(clean_data)

filters.head()

#We can now create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer.

def create_soup(x):
    return x['director'] + ' ' + x['cast'] + ' ' +x['Genre']+' '+ x['description']

filters['soup'] = filters.apply(create_soup, axis=1)

#The next steps are the same as what we did with our plot description based recommender. One important difference is that we use the **CountVectorizer()** instead of TF-IDF.

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filters['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

filters

# Reset index of our main DataFrame and construct reverse mapping as before
filters=filters.reset_index()
indices = pd.Series(filters.index, index=filters['title'])

def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(',',' ').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return net_mov['title'].iloc[movie_indices]

get_recommendations_new('my little pony: a new generation', cosine_sim2)

get_recommendations_new('Supernatural', cosine_sim2)

get_recommendations_new('Naruto', cosine_sim2)
