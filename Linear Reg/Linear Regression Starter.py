import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('CSV/USA_Housing.csv')
# print(df.head())
# sns.pairplot(df)

# plt.show()

# Show distribution on price
sns.distplot(df['Price'])
# plt.show()

sns.heatmap(df.corr())
# plt.show()

# Split data to train and label on

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

# Split data into test and training model
# Get 40% of data and randomly split data base on random_state
x_train, y_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=101)

# Create new Linear Regression model
lm = LinearRegression()

# Fit model with training data
lm.fit(x_train, y_train)

# print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])

print(cdf)
