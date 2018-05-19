"""
Preprocessing of the IMDb dataset.

The raw data is available at "http://ai.stanford.edu/~amaas/data/sentiment/"

The script assumes that the file 'aclImdb_v1.tar.gz' has
been downloaded and extacted into './data/aclImdb'.
"""

from collections import Counter
import glob
import json
import os
from random import shuffle

from gensim.utils import simple_tokenize


def imdb_preprocess():
    imdb_dir = './data/aclImdb'
    subdirs = [
        'train/neg',
        'train/pos',
        'test/neg',
        'test/pos'
    ]

    # Load reviews into memory
    reviews = dict()
    for subdir in subdirs:
        reviews[subdir] = []
        working_dir = os.path.join(imdb_dir, subdir)
        for filepath in glob.glob(working_dir+"/*"):
            with open(filepath, 'r') as f:
                reviews[subdir].append(f.read())

    # Create vocabulary
    vocab_counts = Counter()

    for review_set in [reviews['train/neg'], reviews['train/pos']]:
        for review in review_set:
            vocab_counts.update(simple_tokenize(review))

    word_counts = vocab_counts.most_common()
    word_counts = [pair for pair in word_counts if pair[1] > 2]
    word_index = {
        a[0]: i+2 for i, a in enumerate(word_counts)
    }
    word_index["<EMPTY>"] = 0
    word_index["<UNKNOWN>"] = 1

    with open('./data/word-index.json', 'w') as f:
        json.dump(word_index, f)

    # Encode reviews using word_index
    def word2idx(word):
        if word in word_index.keys():
            return word_index[word]
        else:
            return 1

    def review2idx(review):
        return [word2idx(word) for word in simple_tokenize(review)]

    transformed_reviews = dict()

    for subdir in subdirs:
        transformed_reviews[subdir] = []
        for review in reviews[subdir]:
            transformed_reviews[subdir].append(review2idx(review))

    # Create train, validation and test datasets
    train = [(r, 1) for r in transformed_reviews['train/pos']]
    train += [(r, 0) for r in transformed_reviews['train/neg']]
    shuffle(train)

    test = [(r, 1) for r in transformed_reviews['test/pos']]
    test += [(r, 0) for r in transformed_reviews['test/neg']]
    shuffle(test)

    with open('./data/imdb-reviews.json', 'w') as f:
        json.dump(
            {
                'train': train,
                'test': test
            },
            f
        )


if __name__ == "__main__":
    imdb_preprocess()
