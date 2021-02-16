from keras.models import model_from_json
from keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
from csv import reader
import os
import pandas as pd
import numpy as np

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

new_model = load_model('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\news_sent4.h5')


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = new_model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return predictions


def make_stars(predictions_rate):

    return {
        predictions_rate >= 0 and predictions_rate <= 20:1,
        predictions_rate > 20 and predictions_rate <= 29:2,
        predictions_rate > 30 and predictions_rate <= 50:3,
        predictions_rate > 50 and predictions_rate < 80:4,
        predictions_rate >= 80 and predictions_rate <= 100:5
    }


pred_array = []
star_array = []

# open file in read mode
with open('C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\txt_files\\wonder-woman.csv', 'r', encoding="ISO-8859-1") as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        predictions = sample_predict(row[3]+row[4], pad=True) * 100
        # pred_string = make_stars(predictions)
        pred_array.append(predictions)

for i in pred_array:
    pred_star = make_stars(i.min())
    star_array.append(pred_star)
print("a")

with open("outputWonder.csv", "w") as txt_file:
    for line in star_array:
        txt_file.write(" ".join(str(line[True])) + "\n")

# a = np.asarray(pred_array)
# np.savetxt("pred_model", a, delimiter='\n')
# for sample_text in file_content:
#     pred_array.append(sample_predict(sample_text, pad=True) * 100)



sample_text =  'incredible movie, beautiful message, a super hero being a super hero! Beautiful and exciting'
predictions = sample_predict(sample_text, pad=True) * 100

print('the prob is %.2f' % predictions)


# pred_string = make_stars(predictions.min())
# print(pred_string[1])
# print(predictions.min())


# file_content = []
# dir = "C:\\Users\\invite\\PycharmProjects\\lstm_sentiment\\txt_files"
# for file in os.listdir(dir):
#     if file.endswith(".txt"):
#         with open(os.path.join(dir, file), "r") as fd:
#             file_content.append(fd.read().replace('\n', ''))