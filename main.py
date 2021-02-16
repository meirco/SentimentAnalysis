import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# download the dataset from keras.
from tensorflow.keras.datasets import imdb
# for padding.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# keras do for us word to vec.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

# padding with 0 or truncate data to get maxlen of row = 100
x_train = pad_sequences(x_train, maxlen= 100)
x_test = pad_sequences(x_test, maxlen= 100)

vocab_size = 20000
embed_size = 128

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding


model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_shape=(x_train.shape[1],)))
model.add(LSTM(units=60, activation= 'tanh'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_test, y_test))

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')

tokens = list(sample_text.lower().split())


model.predict()

print("A")

