import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('CSV/titanic_train.csv')

print(train.head())

# Seeing if value is null in data column
print(train.isnull())

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

sns.set_style('whitegrid')

# See who survived, from Titanic data set, split on gender
sns.countplot(x='Survived', hue='Sex', data=train, palette="RdBu_r")
# plt.show()

# See who survived, from Titanic data set, split on cabin class
sns.countplot(x='Survived', hue='Pclass', data=train)
# plt.show()

# See who were on the Titanic, based on Age, dropping null values
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
# plt.show()

# See who were traveling with another person (sibling or spouse), by count, on the ship
sns.countplot(x='SibSp', data=train)
# plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train)
# plt.show()

# Use average of Age for each class for train, if value is null for data point


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
