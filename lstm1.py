
import tensorflow_datasets as tfds
import os.path
import tensorflow as tf
# tfds.disable_progress_bar()
from tensorflow.python.keras.layers import SpatialDropout1D, LSTM

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

buffer_size = 10000
batch_size = 64

padded_shapes = ([None],())

train_dataset = train_dataset.shuffle(10000).padded_batch(64,padded_shapes=padded_shapes)
# train_dataset = tf.random.shuffle(train_dataset)

test_dataset = test_dataset.shuffle(10000).padded_batch(64,padded_shapes=padded_shapes)

# train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64, mask_zero=True),
                             tf.keras.layers.Dropout(0.25),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(32, activation='sigmoid'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(16, activation='tanh'),
                             tf.keras.layers.Dense(1)
                             ])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=14, validation_data=test_dataset, callbacks=[callback], validation_steps=100)


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return predictions


sample_text = 'mefathim is an awsome place, very good'
predictions = sample_predict(sample_text, pad=True) * 100

print('the prob is %.2f' % predictions)



if os.path.isfile('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\news_sent.h5') is False:
    model.save('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\news_sent.h5')

    # import tensorflow_datasets as tfds
    # import torch
    # import os.path
    # import tensorflow as tf
    # from google.colab import drive
    #
    # drive.mount('/content/gdrive')
    # import time
    #
    # # tfds.disable_progress_bar()
    #
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #
    # dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    # train_dataset, test_dataset = dataset['train'], dataset['test']
    #
    # encoder = info.features['text'].encoder
    #
    # buffer_size = 10000
    # batch_size = 64
    #
    # padded_shapes = ([None], ())
    #
    # train_dataset = train_dataset.shuffle(10000).padded_batch(64, padded_shapes=padded_shapes)
    # # train_dataset = tf.random.shuffle(train_dataset)
    #
    # test_dataset = test_dataset.shuffle(10000).padded_batch(64, padded_shapes=padded_shapes)
    #
    # # train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # # test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #
    # # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    # #                               min_delta=0,
    # #                               patience=2,
    # #                               verbose=0, mode='auto')
    #
    # model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64, mask_zero=True),
    #                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #                              tf.keras.layers.Dense(64, activation='relu'),
    #                              tf.keras.layers.Dropout(0.7),
    #                              tf.keras.layers.Dense(32, activation='sigmoid'),
    #                              tf.keras.layers.Dropout(0.2),
    #                              tf.keras.layers.Dense(16, activation='tanh'),
    #                              tf.keras.layers.Dense(1)
    #                              ])
    #
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(1e-4),
    #               metrics=['accuracy'])
    #
    # history = model.fit(train_dataset, epochs=14, validation_data=test_dataset)
    #
    #
    # def pad_to_size(vec, size):
    #     zeros = [0] * (size - len(vec))
    #     vec.extend(zeros)
    #     return vec
    #
    #
    # def sample_predict(sentence, pad):
    #     encoded_sample_pred_text = encoder.encode(sentence)
    #     if pad:
    #         encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    #     encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    #     predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    #
    #     return predictions
    #
    #
    # sample_text = 'mefathim is an awsome place, very good'
    # predictions = sample_predict(sample_text, pad=True) * 100

    # print('the prob is %.2f' % predictions)

    # model_save_name = 'classifier.pt'
    # path = F"/content/gdrive/My Drive/{model_save_name}"
    # model.save(F"/content/gdrive/My Drive/{model_save_name}//news_sent.h5")
    # tf.keras.models.save_model(model, path)
    #
    # # if os.path.isfile('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\news_sent3.h5') is False:
    # #   model.save('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\news_sent3.h5')