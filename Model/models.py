from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_curve, classification_report, roc_auc_score, \
    confusion_matrix
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
from keras.utils.vis_utils import plot_model
import fasttext as f
from tensorflow.python.keras.layers import Flatten

def preproseccingPhase():
    dataset = pd.read_csv('Datasets/tweets_with_hashtag_content.csv')

    dataset.loc[dataset['label'] == 4, 'label'] = 3 #for 3-class representation
    numClasses = 3
    pos_neg_dataset = dataset[['#pos', '#neg']]
    dataset = dataset.drop(['#pos', '#neg'], axis=1)

    dataset['label'].astype('category')

    # One hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(dataset[['label']]).toarray())
    dataset = dataset.join(enc_df)
    dataset = dataset.drop(['label'], axis=1)

    # Load fasttext embeddings
    model = f.load_model('Model/wiki.el.bin')
    embedding_dim = 300

    X_train, X_test, y_train, y_test = train_test_split(dataset['tweet'], dataset.iloc[:, 1:].values,
                                                        test_size=0.1, random_state=42, shuffle=True)
    #pos_neg_train = X_train[['#pos', '#neg']]
    #pos_neg_test = X_test[['#pos', '#neg']]

    # Implement BOG with CountVectorizer
    cv = CountVectorizer()
    preproc = cv.build_analyzer()

    countvecs = cv.fit_transform(X_train)
    vocab = cv.vocabulary_

    maxWords = max_words_in_a_tweet(dataset['tweet'].copy())

    # Create a fill the embedding matrix of our vocabulary
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))

    for word, i in vocab.items():
        if word in model:
            embedding_matrix[i] = model[word]

    embedding_input_train = tweets_to_indices(X_train, preproc, vocab, maxWords)
    embedding_input_test = tweets_to_indices(X_test, preproc, vocab, maxWords)

    return vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train, y_test, enc

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

def statisticsModel(y_pred_lstm, y_pred_mlp, y_test, enc):
    y_pred_lstm = enc.inverse_transform(y_pred_lstm)
    y_pred_mlp = enc.inverse_transform(y_pred_mlp)
    y_test = enc.inverse_transform(y_test)
    accuracy_lstm = metrics.accuracy_score(y_test, y_pred_lstm)
    f1_score_lstm = metrics.f1_score(y_test, y_pred_lstm, average='macro')
    precision_lstm = precision_score(y_test, y_pred_lstm, average='macro')
    recall_lstm = recall_score(y_test, y_pred_lstm, average='macro')

    accuracy_mlp = metrics.accuracy_score(y_test, y_pred_mlp)
    f1_score_mlp = metrics.f1_score(y_test, y_pred_mlp, average='macro')
    precision_mlp = precision_score(y_test, y_pred_mlp, average='macro')
    recall_mlp = recall_score(y_test, y_pred_mlp, average='macro')

    print(confusion_matrix(y_test, y_pred_lstm))
    print(confusion_matrix(y_test, y_pred_mlp))

    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    lstmScores = [round(accuracy_lstm, 2), round(precision_lstm, 2), round(recall_lstm, 2),
                           round(f1_score_lstm, 2)]
    mlpScores = [round(accuracy_mlp, 2), round(precision_mlp, 2), round(recall_mlp, 2), round(f1_score_mlp, 2)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, lstmScores, width, label='LSTM')
    rects2 = ax.bar(x + width / 2, mlpScores, width, label='MLP')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Metrics by model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in rects, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

    print(classification_report(y_test, y_pred_lstm))

def MLP(vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train):
    batch_size = 6
    epochs = 10
    opt = keras.optimizers.Adam(0.0001)
    #BUILDING THE MODEL

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
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='tanh'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(numClasses, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #                      TRAINING THE MODEL
    history = model.fit(embedding_input_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(embedding_input_test, y_test))

    # check_overfitting(history)
    model_json = model.to_json()
    with open("modelMLP.json", "w") as json_file:
        json_file.write(model_json)  # serialize weights to HDF5
    model.save_weights("modelMLP.h5")
    visualizeMLP()
    return model.predict(embedding_input_test)

def LSTMModel(vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train):
    #class_weight = {0: 2.3, 1: 2, 2: 1, 3: 1} #best results
    #class_weight = {0:1.5, 1:5, 2:1}

    trainable = True

    model = Sequential()
    model.add(Embedding(input_dim=len(vocab) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxWords,
                        trainable=trainable))
    model.add((LSTM(20, dropout=0.5)))
    model.add(Dense(numClasses, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(embedding_input_train.shape)
    print(y_train.shape)
    model.fit(embedding_input_train, y_train, epochs=10, batch_size=6, validation_data=(embedding_input_test, y_test))

    y_pred = model.predict(embedding_input_test)
    return y_pred

if __name__ == '__main__':
    vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train, y_test, enc = preproseccingPhase()
    y_pred_lstm = LSTMModel(vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train)
    y_pred_mlp = MLP(vocab, embedding_dim, embedding_matrix, maxWords, embedding_input_train, embedding_input_test, y_train)
    statisticsModel(y_pred_lstm, y_pred_mlp, y_test, enc)
