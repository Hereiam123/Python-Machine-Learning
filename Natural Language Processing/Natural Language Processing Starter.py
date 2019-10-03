import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
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


# print(messages['message'].head(5).apply(text_process))

# Transform into array of occurrences for each word
# and how often it appears in a document
# bow_transformer = CountVectorizer(
#    analyzer=text_process).fit(messages['message'])

# print(len(bow_transformer.vocabulary_))

#mess4 = messages['message'][3]

# print(mess4)

# Get unique words in message selection
#bow4 = bow_transformer.transform([mess4])

# print(bow4)
# print(bow4.shape)

# print(bow_transformer.get_feature_names()[4068])

#messages_bow = bow_transformer.transform(messages['message'])

# print(messages_bow.shape)
#sparsity = (100*messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape))

#tfidf_transformer = TfidfTransformer().fit(messages_bow)
#tfidf4 = tfidf_transformer.transform(bow4)
# print(tfidf4)

# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

#messages_tfidf = tfidf_transformer.transform(messages_bow)

#spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

# print(spam_detect_model.predict(tfidf4)[0])
#all_pred = spam_detect_model.predict(messages_tfidf)
# print(all_pred)

# Run model off pipeline capabilities from scikit learn
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.3)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))
