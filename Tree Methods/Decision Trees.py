import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('CSV/kyphosis.csv')
print(df.head())
sns.pairplot(df, hue="Kyphosis")

# plt.show()

x = df.drop('Kyphosis', axis=1)

y = df['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=101)

dtree = DecisionTreeClassifier()

dtree.fit(x_train, y_train)

predictions = dtree.predict(x_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train, y_train)

rfc_prediction = rfc.predict(x_test)

print(classification_report(y_test, rfc_prediction))
print(confusion_matrix(y_test, rfc_prediction))
