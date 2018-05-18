"""Methods to convert preprocessed IMDb reviews into a tensorflow dataset."""
import json
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import numpy as np


def _filter_review(r):
    return [a for a in r if a > 50]


def _process_input(x, maxlen):
    x_res = [_filter_review(a) for a in x]
    x_res = sequence.pad_sequences(x_res, maxlen=maxlen)
    return x_res


def imdb_datasets(maxlen):
    with open('./data/imdb-reviews.json', 'r') as f:
        imdb = json.load(f)

    train = imdb['train']
    x_train, labels_train = zip(*train)
    labels_train = np.array(labels_train)

    test = imdb['test']
    x_test, labels_test = zip(*test)
    labels_test = np.array(labels_test)

    x_train = _process_input(x_train, maxlen)
    x_test = _process_input(x_test, maxlen)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, labels_train))
        .shuffle(5000)
        .batch(24)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, labels_test))
        .batch(256)
    )

    return train_dataset, test_dataset
