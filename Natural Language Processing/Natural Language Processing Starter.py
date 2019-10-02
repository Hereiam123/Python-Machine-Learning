import nltk
# nltk.download('stopwords')

# Using a collection of text, to detect spam
messages = [line.rstrip() for line in open('CSV/SMSSpamCollection.csv')]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')
