import pandas as pd
import numpy as np

from sklearn import model_selection, feature_extraction, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

dataset = pd.read_csv('annotated.csv')

# Implement BOG with CountVectorizer and TfidfVectorizer
cv = CountVectorizer(ngram_range=(1,1))
tfidf = TfidfVectorizer(smooth_idf=True)
text_counts = cv.fit_transform(dataset['Tweet text']).toarray()
text_counts2 = tfidf.fit_transform(dataset['Tweet text']).toarray()

X_train, X_test, y_train, y_test = train_test_split(text_counts, dataset['Sentiment'], test_size=0.25, random_state=42)

MNB = MultinomialNB()
MNB.fit(X_train, y_train)

y_pred = MNB.predict(X_test)

print('Accuracy score of NB = ', metrics.accuracy_score(y_test, y_pred))