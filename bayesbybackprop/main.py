import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys

import numpy as np

from bayesnn.bayesdense import gaussian_mixture_density
from bayesnn.model import BayesDenseNet, simple_model
from bayesnn.train_loop import train_loop
from bayesnn.pruning import prune_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pi",
        help="mixture coefficient",
        default='0.25',
        type=float
    )
    parser.add_argument(
        "--log_s1",
        default='0',
        type=float
    )
    parser.add_argument(
        "--log_s2",
        default='-2',
        type=float
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
        default='model'
    )
    args = parser.parse_args(sys.argv[1:])

    model = BayesDenseNet(
        1200,
        args.pi,
        args.log_s1,
        args.log_s2
    )
    if args.epochs:
        train_loop(
            model,
            args.logdir,
            args.epochs
        )
    prune_eval(model, args.logdir)
