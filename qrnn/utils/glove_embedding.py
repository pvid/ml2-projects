"""
GloVe embedding processing for IMDb vocabulary.

The script generates GloVe embedding matrix for the IMDb dataset preprocessed
by 'imdb_preprocess.py'. It checks if, 'glove.840B.300d.txt' is present and
downloads it if it is not. The file can be found on
"https://nlp.stanford.edu/projects/glove/".

It is recommended to download the file separately, especially for anxious
or impatient people, because we did not implement a progress bar.
"""
import os
import urllib.request
import json
import zipfile

import numpy as np


def download_glove():
    """Download and unzip GloVe embedding file."""
    glove_path = './data/glove.840B.300d.txt'
    glove_path_zip = './data/glove.840B.300d.zip'
    glove_url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"

    if os.path.exists(glove_path):
        pass
    elif os.path.exists(glove_path_zip):
        print("The file 'glove.840B.300d.zip' found. Unzipping...")
        with zipfile.ZipFile(glove_path_zip, "r") as zip:
            zip.extractall("./data/")
        print("Done!")
    else:
        print("Downloading pretrained GloVe vectors... it will take some time")
        urllib.request.urlretrieve(glove_url, glove_path_zip)
        print("Unzipping...")
        with zipfile.ZipFile(glove_path_zip, "r") as zip:
            zip.extractall("./data/")
        print("Done!")


def generate_embedding(glove_path='./data/glove.840B.300d.txt'):
    """
    Read GloVe file and generate embedding.

    Parameters
    ----------

    glove_path:
        Path to a GloVe embedding .txt file.
    """
    print("Generating embedding matrix for IMDb dataset...")
    with open('./data/word-index.json', 'r') as f:
        word_index = json.load(f)
    n_words = len(word_index)
    embed_weights = np.zeros((n_words, 300), np.float32)

    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            items = line.split(sep=' ')
            word = items[0]
            if word in word_index.keys():
                vector = np.array(items[1:301], np.float32)
                idx = word_index[word]
                embed_weights[idx, :] = vector
    np.save("./data/embedding.npy", embed_weights)
    print("Done!")


if __name__ == "__main__":
    download_glove()
    generate_embedding()
