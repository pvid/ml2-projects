# Weight Uncertainty & Bayes by Backprop

A TensorFlow implementation of a variational bayesian dense neural network
based on  ["Weight Uncertainty in Neural Networks"](https://arxiv.org/pdf/1505.05424.pdf)

We focus on the use of posterior distribution of weights for pruning
(reducing the number of weighs). For details, see my [report](../bayesbybackprop/report.pdf).

The implementation of the bayesian dense layes is in
[bayesnn/bayesdense.py](../bayesbybackprop/bayesnn/bayesdense.py)

## Requirements

Tensorflow == 1.7.0

### How to run training and evaluation

To run pruning evaluation on a pretrained model:

    $ python main.py --logdir model_pretrained --epochs 0

To train a model with different parameters

    $ python main.py --logdir logdir --log_s1 value --log_s2 value --pi value --epochs n_epochs

The default values are the following:
logdir: model
log_s1: 0
log_s2: -2
epochs: 10


### LaTex template

The template used can be found [here](http://www.latextemplates.com/template/journal-article).
