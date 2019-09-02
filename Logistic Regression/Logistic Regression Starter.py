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
