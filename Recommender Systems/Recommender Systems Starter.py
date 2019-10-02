import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

# Movie lens recommendation data set
df = pd.read_csv('CSV/u.data.csv', sep="\t", names=columns_names)

# print(df.head())

movie_titles = pd.read_csv("CSV/Movie_Id_Titles.csv")

# print(movie_titles.head())

# Merge item id and movie title

df = pd.merge(df, movie_titles, on='item_id')

# print(df.head())

sns.set_style('white')

# Average ratings for movie
# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

# Sort by num of ratings
# print(df.groupby('title')['rating'].count(
# ).sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
# print(ratings.head())
# ratings['num of ratings'].hist(bins=70)
# plt.show()

# ratings['rating'].hist(bins=70)
# plt.show()

# Compare ratings to num of ratings
# sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
# plt.show()

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# print(moviemat.head())

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

# Correlation of other movies to Star Wars rating
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
# print(corr_starwars.head())

# Correlation of other user reviews for star wars
# print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

# More convincing correlation
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head())

"""Liar Liar"""
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head())
