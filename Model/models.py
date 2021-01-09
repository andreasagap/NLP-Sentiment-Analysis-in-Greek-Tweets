import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_curve, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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


def statisticsModel(y_pred):
    y_pred = enc.inverse_transform(y_pred)

    print('Accuracy model: ', metrics.accuracy_score(y_test, y_pred))
    print('F1 score model: ', metrics.f1_score(y_test, y_pred, average='micro'))
    print(model.summary())
    print("Precision: " + str(precision_score(y_test, y_pred)))
    print("Recall " + str(recall_score(y_test, y_pred)))
    fpr, tpr, roc = np.array(roc_curve(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Area under the curve: " + str(roc_auc))
    print()

    print(classification_report(y_test, y_pred))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

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
    batch_size = 64
    epochs = 90
    opt = keras.optimizers.Nadam(0.0001)
    #                      BUILDING THE MODEL
    model = Sequential()
    #model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
    # model.add(Dense(100, activation='relu', ))
    # model.add(Dropout(0.8))
    model.add(Embedding(input_dim=len(vocab) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxWords,
                        trainable=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.7))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #                      TRAINING THE MODEL
    history = model.fit(embedding_input_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(embedding_input_test, y_test)
                        )

    check_overfitting(history)
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


dataset = pd.read_csv('annotated.csv')
dataset['Sentiment'].astype('category')

enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(dataset[['Sentiment']]).toarray())
dataset = dataset.join(enc_df)
dataset = dataset.drop(['Sentiment'], axis=1)

model = f.load_model('wiki.el.bin')
embedding_dim = 300

# Implement BOG with CountVectorizer and TfidfVectorizer
cv = CountVectorizer(ngram_range=(1, 1))
tfidf = TfidfVectorizer(smooth_idf=True)

X_train, X_test, y_train, y_test = train_test_split(dataset['Tweet text'], dataset.iloc[:, 1:].values, test_size=0.25,
                                                    random_state=42)

# text_counts = cv.fit_transform(dataset['Tweet text']).toarray()
# text_counts2 = tfidf.fit_transform(dataset['Tweet text']).toarray()

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

statisticsModel(LSTM(vocab, embedding_dim, embedding_matrix, maxWords))
statisticsModel(MLP(vocab, embedding_dim, embedding_matrix, maxWords))


