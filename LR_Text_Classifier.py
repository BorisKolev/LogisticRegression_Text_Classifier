from bs4 import BeautifulSoup
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import pandas as pd
import nltk
import re

# ------------------ Reading the data set ------------------

dataFrame = pd.read_csv('IMDB Dataset.csv')

# Shuffle data set so it is less biased.
dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

# ------------------ Pre-processing ------------------

t = time()

# Converting all words to lower case from review column
dataFrame['review'] = dataFrame['review'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

# Removing english stop words from review column
english_stop_words = stopwords.words("english")
dataFrame['review'] = dataFrame['review'].apply(
    lambda x: " ".join([x for x in x.split() if x not in english_stop_words]))

# Removing non-alpha characters from review column
dataFrame['review'] = dataFrame['review'].apply(
    lambda x: " ".join([re.sub('[^A-Za-z]+', '', x) for x in nltk.word_tokenize(x)]))

# Remove html tags from review column
dataFrame['review'] = dataFrame['review'].apply(lambda x: BeautifulSoup(x, "html5lib").get_text())

print('Time to pre-process all the data: {} seconds'.format(round((time() - t), 2)))

# ------------------ Feature Extraction ------------------

t = time()

# splitting data into 4000 rows training and the last 1000 rows testing sets.
review_train, review_test, sentiment_train, sentiment_test = train_test_split(dataFrame["review"],
                                                                              dataFrame["sentiment"],
                                                                              test_size=0.20,
                                                                              shuffle=False)
# print("Train: ", review_train.shape, sentiment_train.shape, "Test: ", (review_test.shape, sentiment_test.shape))

Tf_idf_V = TfidfVectorizer()
tf_review_train = Tf_idf_V.fit_transform(review_train)
tf_review_test = Tf_idf_V.transform(review_test)

print('Time to split into training sets and calculate TF-IDF : {} seconds'.format(round(time() - t), 2))

# ------------------ Logistic Regression Classifier ------------------

t = time()

classifier = LogisticRegression(max_iter=1000)
classifier.fit(tf_review_train, sentiment_train)
sentiment_prediction = classifier.predict(tf_review_test)
report = classification_report(sentiment_test, sentiment_prediction)

print('Time to train and predict: {} seconds'.format(round(time() - t), 2))

print("\nReport ------------------------------------------------------ ")
print(report)
