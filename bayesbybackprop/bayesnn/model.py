import tensorflow as tf
import numpy as np

from .bayesdense import gaussian_mixture_density, BayesianDense

def simple_model(x, sample=False):
    prior = gaussian_mixture_density(0.25, np.exp(-1), np.exp(-6))
    x = BayesianDense(1200, activation=tf.nn.relu, kernel_prior_density=prior)(x, sample)
    x = BayesianDense(1200, activation=tf.nn.relu, kernel_prior_density=prior)(x, sample)
    out = BayesianDense(10)(x, sample)
    return out

class BayesDenseNet(object):

    def __init__(self, n_units=1200, pi=0.25, log_s1=0.0, log_s2=-2):
        # prior = gaussian_mixture_density(pi, math.exp(log_s1), math.exp(log_s2))
        prior = gaussian_mixture_density(0.25, np.exp(-1), np.exp(-6))
        self.layers = []
        self.layers.append(
            BayesianDense(
                n_units,
                activation=tf.nn.relu,
                kernel_prior_density=prior
            )
        )
        self.layers.append(
            BayesianDense(
                n_units,
                activation=tf.nn.relu,
                kernel_prior_density=prior
            )
        )
        self.layers.append(
            BayesianDense(
                10,
                kernel_prior_density=prior
            )
        )


    def __call__(self, input, sample):
        x = self.layers[0](input, sample)
        x = self.layers[1](x, sample)
        out = self.layers[2](x, sample)

        self.complexity_cost = self.layers[0].complexity_cost
        self.complexity_cost += self.layers[1].complexity_cost
        self.complexity_cost += self.layers[2].complexity_cost
        self.means = []
        self.sigmas = []
        for layer in self.layers:
            self.means.append(layer.kernel_mean)
            self.means.append(layer.bias_mean)

            self.sigmas.append(layer.kernel_sigma)
            self.sigmas.append(layer.bias_sigma)

        return out

    def random_prune(self, proportion):
        ops = []
        for m in self.means:
            ops.append(_random_prune_one(m, proportion))
        return tf.group(ops)

    def absolute_value_prune(self, threshold):
        ops = []
        for m in self.means:
            ops.append(_absolute_value_prune_one(m, threshold))
        return tf.group(ops)

    def signal2noise_prune(self, threshold):
        ops = []
        for m, s in zip(self.means, self.sigmas):
            ops.append(_signal2noise_prune_one(m, s, threshold))
        return tf.group(ops)


def _random_prune_one(tensor, proportion):
    rand = tf.random_uniform(
        shape=tensor.shape,
        dtype=tf.float64
    )
    changed = tf.where(
        rand < proportion/100,
        tf.zeros(shape=tensor.shape, dtype=tf.float64),
        tensor)
    return tf.assign(tensor, changed)


def _absolute_value_prune_one(tensor, threshold):
    changed = tf.where(
        tf.abs(tensor) < threshold,
        tf.zeros(shape=tensor.shape, dtype=tf.float64),
        tensor)
    return tf.assign(tensor, changed)


def _signal2noise_prune_one(mean, sigma, threshold):
    changed = tf.where(
        tf.abs(mean)/sigma < threshold,
        tf.zeros(shape=mean.shape, dtype=tf.float64),
        mean)
    return tf.assign(mean, changed)
