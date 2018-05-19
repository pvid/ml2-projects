import argparse
import os
import sys

import numpy as np

from qr_imdb.models import model_lstm, model_cudnnlstm, model_qrnn, \
    model_qrnn_alt
from qr_imdb.train_eval import train_eval_loop
from utils.glove_embedding import download_glove, generate_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glovepath",
        help="""path to the GloVe embedding txt file.
            it is needed only when run for the first time
            to generate word embedding matrix.""",
        default='./data/glove.840B.300d.txt'
    )
    parser.add_argument(
        "--model",
        help="which NN architecture to use",
        choices=['qrnn', 'qrnn_alt', 'lstm', 'cudnnlstm'],
        default='qrnn'
    )
    parser.add_argument(
        "--epochs",
        help="number of epoch to train for",
        default='10',
        type=int
    )
    parser.add_argument(
        "--logdir",
        help="directory used for summaries and saves",
    )
    args = parser.parse_args(sys.argv[1:])

    if args.logdir is None:
        args.logdir = 'model_' + args.model

    if not os.path.exists("./data/embedding.npy"):
        if args.glovepath is None:
            if not os.path.exists('./data/glove.840B.300d.txt'):
                download_glove()
            generate_embedding('./data/glove.840B.300d.txt')
        else:
            generate_embedding(args.glovepath)
    embed_weights = np.load("./data/embedding.npy")
    print("Embedding shape is: {}".format(embed_weights.shape))

    model_map = {
        'qrnn': model_qrnn,
        'qrnn_alt': model_qrnn_alt,
        'lstm': model_lstm,
        'cudnnlstm': model_cudnnlstm
    }
    train_eval_loop(
        args.epochs,
        embed_weights,
        model_map[args.model],
        args.logdir)
