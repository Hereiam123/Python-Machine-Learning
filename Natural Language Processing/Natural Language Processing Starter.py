import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

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
#print(messages[messages['length'] == 910]['message'].iloc[0])

# Returned hist shows that spam messages tend to have longer word counts
#messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

# plt.show()

mess = 'Sample message! Notice: its has punctuation'
no_punctuation = [c for c in mess if c not in string.punctuation]

# print(no_punctuation)

# Filtering on common words that won't be important
# for filtering spam against
stopwords.words('english')

no_punctuation = ''.join(no_punctuation)

clean_meass = [word for word in no_punctuation.split(
) if word.lower() not in stopwords.words('english')]

# print(clean_meass)


def text_process(mess):
    """
    1. remove punctuation
    2.remove stop words
    3. retrun list of clean text words
    """

    no_punctuation = [char for char in mess if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


print(messages['message'].head(5).apply(text_process))
