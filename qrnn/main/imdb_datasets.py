import json
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import numpy as np


def process_input(x, maxlen):
    x_res = [a[1:] for a in x]
    end_index = [len(a)-1 if len(a) <= maxlen else maxlen-1 for a in x_res]
    x_res = sequence.pad_sequences(x_res, maxlen=maxlen)
    return x_res, np.array(end_index)


def imdb_datasets(maxlen):
    with open('./data/imdb_reviews.json', 'r') as f:
        imdb = json.load(f)

    train = imdb['train']
    x_train, labels_train = zip(*train)
    labels_train = np.array(labels_train)

    validation = imdb['validation']
    x_validation, labels_validation = zip(*validation)
    labels_validation = np.array(labels_validation)

    test = imdb['test']
    x_test, labels_test = zip(*test)
    labels_test = np.array(labels_test)

    x_train, train_end_idx = process_input(x_train, maxlen)
    x_validation, validation_end_idx = process_input(x_validation, maxlen)
    x_test, test_end_idx = process_input(x_test, maxlen)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, train_end_idx, labels_train)
    ).shuffle(500)
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (x_validation, validation_end_idx, labels_validation)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, test_end_idx, labels_test)
    )

    return train_dataset, validation_dataset, test_dataset
