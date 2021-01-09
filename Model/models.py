from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_curve, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import keras.backend as K
from itertools import product
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import fasttext as f
from tensorflow.python.keras.layers import Flatten


def max_words_in_a_tweet(tweets):
    max = 0
    for i, tweet in enumerate(tweets):
        tweet_len = len(tweet.split())
        if (tweet_len) > max:
            max = tweet_len
    return max


def tweets_to_indices(tweets, preproc, vocab, maxWords):
    new_tweets = []  # tweets with indices
    for tweet in tweets:
        tweet = preproc(tweet)
        new = []
        for w in tweet:
            if w in vocab:
                new.append(vocab[w])
            else:
                continue
        new_tweets.append(new)
    return pad_sequences(new_tweets, maxlen=maxWords)


def statisticsModel(y_pred, y_test):
    y_pred = enc.inverse_transform(y_pred)
    y_test = enc.inverse_transform(y_test)
    print('Accuracy model: ', metrics.accuracy_score(y_test, y_pred))
    print('F1 score model: ', metrics.f1_score(y_test, y_pred, average='macro'))

    print("Precision: " + str(precision_score(y_test, y_pred, average='macro')))
    print("Recall " + str(recall_score(y_test, y_pred, average='macro')))

    print(classification_report(y_test, y_pred))


def check_overfitting(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def MLP(vocab, embedding_dim, embedding_matrix, maxWords):
    batch_size = 16
    epochs = 40
    opt = keras.optimizers.Adam(0.0001)
    #                      BUILDING THE MODEL

    model = Sequential()
    # model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
    # model.add(Dense(100, activation='relu', ))
    # model.add(Dropout(0.8))
    model.add(Embedding(input_dim=len(vocab) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxWords,
                        trainable=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='tanh'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #                      TRAINING THE MODEL
    history = model.fit(embedding_input_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(embedding_input_test, y_test),
                        class_weight={0: 0.9, 1: 1.1, 2: 1, 3: 0.8})

    # check_overfitting(history)
    return model.predict(embedding_input_test)


def LSTM(vocab, embedding_dim, embedding_matrix, maxWords):
    trainable = True
    opt = keras.optimizers.RMSprop()

    model = Sequential()
    model.add(Embedding(input_dim=len(vocab) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxWords,
                        trainable=trainable))
    model.add(Bidirectional(LSTM(10, dropout=0.3)))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(embedding_input_train, y_train, epochs=10, batch_size=250)

    return model.predict(embedding_input_test)


# usecols=[1,2],names=["Tweet","Sentiment"]
dataset = pd.read_csv('tweets_preprocessed.csv', usecols=[0, 1], names=["tweet", "label"], header=None)
dataset = dataset[dataset["label"].notna()]
dataset = dataset[dataset["label"] != 10.0]
dataset['label'].astype('category')
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(dataset[['label']]).toarray())
dataset = dataset.reset_index(drop=True)
dataset = dataset.join(enc_df)
dataset = dataset.drop(['label'], axis=1)
# dataset = dataset[dataset.iloc[:, 1].notna()]
model = f.load_model('/Users/Andreas/Desktop/NLP-AUTH/wiki.el.bin')
embedding_dim = 300

# Implement BOG with CountVectorizer and TfidfVectorizer
cv = CountVectorizer(ngram_range=(1, 1))
tfidf = TfidfVectorizer(smooth_idf=True)
X_train, X_test, y_train, y_test = train_test_split(dataset['tweet'], dataset.iloc[:, 1:].values, test_size=0.25,
                                                    random_state=0)

# text_counts = cv.fit_transform(dataset['Tweet text']).toarray()
# text_counts2 = tfidf.fit_transform(dataset['Tweet text']).toarray()

preproc = cv.build_analyzer()
countvecs = cv.fit_transform(X_train)
vocab = cv.vocabulary_
features = cv.get_feature_names()

maxWords = max_words_in_a_tweet(dataset['tweet'].copy())

# Create a fill the embedding matrix of our vocabulary
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))

for word, i in vocab.items():
    if word in model:
        embedding_matrix[i] = model[word]

embedding_input_train = tweets_to_indices(X_train, preproc, vocab, maxWords)
embedding_input_test = tweets_to_indices(X_test, preproc, vocab, maxWords)

feature_shape = embedding_dim

# MODEL declaration

# statisticsModel(LSTM(vocab, embedding_dim, embedding_matrix, maxWords))
statisticsModel(MLP(vocab, embedding_dim, embedding_matrix, maxWords), y_test)
