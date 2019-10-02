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
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

# Sort by num of ratings
print(df.groupby('title')['rating'].count(
).sort_values(ascending=False).head())
