import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

ad_data = pd.read_csv('CSV/advertising.csv')

print(ad_data.head())

print(ad_data.describe())

ad_data['Age'].plot.hist(bins=30)

# plt.show()

sns.jointplot(x="Age", y="Area Income", data=ad_data)

# plt.show()

sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde")

# plt.show()

sns.jointplot(x="Daily Time Spent on Site",
              y="Daily Internet Usage", data=ad_data)

# plt.show()

sns.pairplot(ad_data)

# plt.show()

x = ad_data[['Daily Time Spent on Site', 'Age',
             'Area Income', 'Daily Internet Usage', 'Male']]

y = ad_data['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=101)

# Fit for Logistic Regression model
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Predictions
predictions = logmodel.predict(x_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
