"""
The script generates GloVe embedding matrix for the IMDb dataset preprocessed by
'imdb_preprocess.py'.

It checks if, 'glove.840B.300d.txt' is present and downloads it if it is not.
The file can be found on "https://nlp.stanford.edu/projects/glove/". It is recommended
to download the file separately, especially for anxious or impatient people, because
we did not implement a progress bar.
"""


import logging
import os
import urllib.request
import sys
import json

import numpy as np
import zipfile

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

glove_path = '../data/glove.840B.300d.txt'
glove_path_zip = '../data/glove.840B.300d.zip'
glove_url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"


def download_glove():
    if os.path.exists(glove_path):
        logger.info("The file 'glove.840B.300d.txt' found.")
    elif os.path.exists(glove_path_zip):
        logger.info("The file 'glove.840B.300d.zip' found. Unzipping...")
        with zipfile.ZipFile(glove_path_zip, "r") as zip:
            zip.extractall("../data/")
        logger.info("Done!")
    else:
        logger.info("Downloading pretrained GloVe vectors... it will take some time")
        urllib.request.urlretrieve(glove_url, glove_path_zip)
        logger.info("Unzipping...")
        with zipfile.ZipFile(glove_path_zip, "r") as zip:
            zip.extractall("../data/")
        logger.info("Done!")


def generate_embedding():
    logger.info("Generating embedding matrix for IMDb dataset...")
    with open('../data/word-index.json', 'r') as f:
        word_index = json.load(f)
    n_words = len(word_index)
    embed_weights = np.zeros((n_words, 300), np.float32)

    with open('../data/glove.840B.300d.txt', 'r', encoding="utf-8") as f:
        for line in f:
            items = line.split(sep=' ')
            word = items[0]
            if word in word_index.keys():
                vector = np.array(items[1:301], np.float32)
                idx = word_index[word]
                embed_weights[idx,:] = vector
    np.save("../data/embedding.npy",embed_weights)
    logger.info("Done!")

if __name__ == "__main__":
    download_glove()
    generate_embedding()
