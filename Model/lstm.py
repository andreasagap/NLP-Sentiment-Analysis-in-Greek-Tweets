import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, feature_extraction, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import class_weight

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import fasttext as f

def max_words_in_a_tweet(tweets):
    max=0
    for i,tweet in enumerate(tweets):
        tweet_len = len(tweet.split())
        if (tweet_len)> max:
            max=tweet_len
    return max

def tweets_to_indices(tweets,preproc,vocab,maxWords):
    new_tweets=[]  #tweets with indices
    for tweet in tweets:
        tweet=preproc(tweet)
        new=[]
        for w in tweet:
            if w in vocab:
                new.append(vocab[w])
            else:
                continue
        new_tweets.append(new)
    return pad_sequences(new_tweets, maxlen=maxWords)

def tweets_to_indices(tweets,preproc,vocab,maxWords):
    new_tweets=[]  #tweets with indices
    for tweet in tweets:
        tweet=preproc(tweet)
        new=[]
        for w in tweet:
            if w in vocab:
                new.append(vocab[w])
            else:
                continue
        new_tweets.append(new)
    return pad_sequences(new_tweets, maxlen=maxWords)



dataset = pd.read_csv('tweets_preprocessed.csv')
dataset['Sentiment'].astype('category')

enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(dataset[['Sentiment']]).toarray())
dataset = dataset.join(enc_df)
dataset = dataset.drop(['Sentiment'], axis=1)

model = f.load_model('wiki.el.bin')
embedding_dim = 300

# Implement BOG with CountVectorizer and TfidfVectorizer
cv = CountVectorizer(ngram_range=(1,1))
tfidf = TfidfVectorizer(smooth_idf=True)

X_train, X_test, y_train, y_test = train_test_split(dataset['Tweet text'], dataset.iloc[:, 1:].values, test_size=0.25, random_state=42, shuffle=True)

#text_counts = cv.fit_transform(dataset['Tweet text']).toarray()
#text_counts2 = tfidf.fit_transform(dataset['Tweet text']).toarray()

preproc = cv.build_analyzer()
countvecs = cv.fit_transform(X_train)
vocab = cv.vocabulary_
features = cv.get_feature_names()

maxWords = max_words_in_a_tweet(dataset['Tweet text'].copy())

# Create a fill the embedding matrix of our vocabulary
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))

for word, i in vocab.items():
    if word in model:
        embedding_matrix[i] = model[word]

embedding_input_train = tweets_to_indices(X_train, preproc, vocab, maxWords)
embedding_input_test = tweets_to_indices(X_test, preproc, vocab, maxWords)

feature_shape = embedding_dim

# MODEL declaration
class_weight = {0: 1, 1: 4, 2:1, 3:1}

trainable = True
opt = keras.optimizers.RMSprop()

model = Sequential()
model.add(Embedding(input_dim=len(vocab) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxWords,
                    trainable=trainable))
model.add(Bidirectional(LSTM(10, dropout=0.4, return_sequences=True)))
model.add((LSTM(20, dropout=0.7)))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(embedding_input_train, y_train, epochs=10, batch_size=250, class_weight=class_weight)

y_pred = model.predict(embedding_input_test)

y_pred = enc.inverse_transform(y_pred)

print('LSTM Accuracy model: ', metrics.accuracy_score(y_test, y_pred))
print('LSTM F1 score model: ', metrics.f1_score(y_test, y_pred, average='micro'))
print('LSTM Recall model: ', metrics.recall_score(y_test, y_pred, average='micro'))
print('LSTM Precision score model: ', metrics.precision_score(y_test, y_pred, average='micro'))

for i in range(len(y_train))
len(y_train[ι])
