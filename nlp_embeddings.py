import pandas as pd
import numpy as np
import codecs

from sklearn import model_selection, feature_extraction, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, MaxPool1D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('annotated.csv').values

X=dataset[:,0]
Y=dataset[:,1]
Y = np.asarray(Y).astype('float32')

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_test = tokenizer.texts_to_sequences(X_test)

maxlen=100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)


emb_dim=100
vocab=len(tokenizer.word_index)+1
emb_mat= np.zeros((vocab,emb_dim))
#Initializing a zero matrix for each word, they will be compared to have their final embedding

with codecs.open('model.txt', 'r', encoding='utf-8', errors='ignore') as f:
  for line in f:
    word, *emb = line.split() 
    if word in tokenizer.word_index:
        ind=tokenizer.word_index[word]
        emb_mat[ind]=np.array(emb,dtype="float32")[:emb_dim]
        

emb_dim=100
maxlen=100
model= Sequential()
model.add(Embedding(input_dim=vocab, output_dim=emb_dim,weights=[emb_mat], input_length=maxlen,trainable=False))
model.add(MaxPool1D())
model.add(Dense(16,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()