import nltk
# nltk.download('stopwords')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using a collection of text, to detect spam
messages = [line.rstrip() for line in open('CSV/SMSSpamCollection.csv')]

"""for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')"""

messages = pd.read_csv('CSV/SMSSpamCollection.csv',
                       sep='\t', names=['label', 'message'])

# print(messages.head())

# print(messages.describe())

# print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)

# print(messages.head())

# messages['length'].plot.hist(bins=50)

# plt.show()

# print(messages['length'].describe())

# Getting longest message length to check out
print(messages[messages['length'] == 910]['message'].iloc[0])

# Returned hist shows that spam messages tend to have longer word counts
messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

plt.show()
