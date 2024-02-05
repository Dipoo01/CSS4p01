# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:02:50 2024

@author: Dipoo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Dipoo/Desktop/Movie_dataset/movie_dataset.csv")
pd.set_option('display.max_rows',None)

print(df)

x = df["Revenue (Millions)"].mean()
df["Revenue (Millions)"].fillna(x, inplace = True)

print(df)

x = df["Metascore"].mean()
df["Metascore"].fillna(x, inplace = True)

print(df)

df.to_csv("C:/Users/Dipoo/Desktop/Movie_dataset/movie_dataset_cleaned.csv")

import matplotlib.pyplot as plt

df.info()

# create a dictionary
# key = old name
# value = new name

dict = {'Runtime (Minutes)': 'Runtime', 'Revenue (Millions)': 'Revenue'}
print("\nAfter rename")

df.rename(columns=dict,inplace=True)

display(df)

selected_columns =['Year','Runtime','Rating','Votes','Revenue','Metascore']
selected_data = df[selected_columns]
correlation_matrix = selected_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot =True,cmap = 'crest',fmt='.2f',linewidth=0.5)
plt.title('Correlation Matrix')
plt.show()


# arrange the df by ratings
df=df.sort_values(by='Rating',ascending=False)
top_movies = df[['Rating']].head(10)
print(top_movies)

#arrange the df by year of release
df=df.sort_values(by='Year',ascending=False)
Year_of_release = df[['Year']].head
print(Year_of_release)

# arrange the df by director
df=df.sort_values(by='Director',ascending=True)
Director_con = df[['Director']]



import pandas as pd
df = pd.read_csv("C:/Users/Dipoo/Desktop/Movie_dataset/movie_dataset_cleaned.csv")
print(df)
print(df.describe())
"""
df= [{'Year', 'Rating'}]
year_ratings = df
for item in df:
    Year = item['Year']
    Rating = item['Rating']

if Year in year_ratings:
    year_ratings['Year'].append('Rating')
else:
    year_ratings[Year] = [Rating]
    
highest_avg_rating = 0
year_with_highest_rating = None

for Year, ratings in year_ratings.items():
    avg_rating = sum(Rating) / len(Rating)
    
    if avg_rating > highest_avg_rating:
        highest_avg_rating = avg_rating
        year_with_highest_rating = Year

print(f"The year with the highest average rating is: {year_with_highest_rating}")
"""
"""
# frequency of movies per year

df=df.sort_values(by='Year',ascending=False)
df['Year'].plot(kind='hist', bins=10)

df=df.sort_values(by='Rating',ascending=False)
"""

#unique values in all the columns

df.nunique()
df.dtypes

#number of movies directed by each director
df.Director.value_counts().head(5)

#runtime
df['Runtime (Minutes)'].describe()
df['Runtime (Minutes)'].plot.hist()
runtime_bin_edges = [66,100,111,123,187]
runtime_bin_names = ['Short','Medium','Moderately Long','Long']
df['Runtime_level'] = pd.cut(df['Runtime (Minutes)'], bins = runtime_bin_edges,labels=runtime_bin_names)
df.head()
"""
ax = plt.subplots(1, figsize=(10,8))
df['Runtime (Minutes)'].value_counts().plot.bar(color=['blue','orange','green','red','yellow'])
ax.set_ylabel('Votes')
ax.set_xlabel('Runtime_level')
ax.set_xticklabels(df.Runtime_level.value_counts().index, rotation = 45);
ax.set_title('Movies Count of each Runtime level', y=1.02)
df['Runtime_level']
"""
"""
run_stats = df.groupby('Runtime_level')['Revenue (Millions)', 'Rating','Metascore'].mean()
run_stats
"""

#Revenue per year stats
"""
revenue_year = df.groupby('Year')['Revenue (Millions)'].sum()
revenue_year
plt.figure(figsize=(10,10))
revenue_year.plot()
plt.ylabel('Revenue')

average_rev = df.groupby('Year')['Revenue (Millions)'].Mean()
plt.figure(figsize=(10,10))
average_rev.plot()
plt.ylabel('Revenue')

"""
"""
#Directors who are active producing more than 5 movies
Active_directors = df['Director'].value_counts()[df['Director'].value_counts() >=5]
Active.director_movies = df[df.Director.isin(Active_directors.index.tolist(()]
Active_director_movies.head()

Active_director_stats = Active_director_movies.groupby('Director')['Revenue (Millions)', 'Rating', 'Metascore'].mean().sort_values('Revenue (Millions)'),ascending=False)
Active_director_stats[:5]

plt.figure(figsize=(10,10))
Active_director_stats[:5]['Revenue (Millions)'].plot.barh()
plt.xlabel('Revenue')
"""


