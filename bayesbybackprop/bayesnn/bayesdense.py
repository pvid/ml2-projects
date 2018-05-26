"""
A TensorFlow implementation based on the paper "Weight Uncertainty
in Neural Networks" (https://arxiv.org/pdf/1505.05424.pdf).

The implementation is adapted directly from the Dense layer implementation
of TensorFlow 1.7.0 (https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/layers/core.py)
"""


from math import pi
import math

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.distributions import normal


def normal_pdf(x, mean, sigma):
    result = -0.5*gen_math_ops.square((x-mean)/sigma)
    result = gen_math_ops.exp(result)
    result /= (2*pi)**0.5
    result /= sigma
    return result


def gaussian_mixture_density(proportion, sigma1, sigma2):
    if proportion == 1:
        def density(x):
            return normal_pdf(x, 0, sigma1)
        return density
    else:
        def density(x):
            res = proportion*normal_pdf(x, 0, sigma1)
            res += (1-proportion)*normal_pdf(x, 0, sigma2)
            return res
        return density


def _complexity_cost(weights, mean, sigma, prior_density):
    k = -0.5*math.log(2*math.pi)
    cost = k - gen_math_ops.log(sigma) - 0.5*gen_math_ops.square((weights-mean)/sigma)
    cost -= gen_math_ops.log(prior_density(weights))
    cost = math_ops.reduce_sum(cost)
    return cost


class BayesianDense(base.Layer):
    """
    Dense layer for Bayes by Backprop

    The use of sampled weights (for training and posterior distibution
    evaluation) if contolled by the 'sample' argument of the 'call' method.
    """
    def __init__(
            self, units,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=None,
            kernel_rho_initializer=init_ops.constant_initializer(-3),
            bias_rho_initializer=init_ops.constant_initializer(-3),
            kernel_prior_density=gaussian_mixture_density(1, 0.1, 0),
            bias_prior_density=gaussian_mixture_density(1, 0.1, 0),
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            **kwargs):
        super(BayesianDense, self).__init__(
            trainable=trainable, name=name,
            activity_regularizer=activity_regularizer,
            **kwargs
        )
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_rho_initializer = kernel_rho_initializer
        self.bias_rho_initializer = bias_rho_initializer
        self.kernel_prior_density = kernel_prior_density
        self.bias_prior_density = bias_prior_density
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel_mean = self.add_variable(
            'kernel_mean',
            shape=[input_shape[-1].value,
                   self.units],
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        self.kernel_rho = self.add_variable(
            'kernel_rho',
            shape=[input_shape[-1].value,
                   self.units],
            initializer=self.kernel_rho_initializer,
            dtype=self.dtype,
            trainable=True
        )
        self.kernel_sigma = nn.softplus(self.kernel_rho)
        kernel_gaussian = random_ops.random_normal(
            name="kernel_gaussian",
            shape=[input_shape[-1].value,
                   self.units],
            dtype=self.dtype
            )
        self.kernel_sample = (
            self.kernel_mean
            + self.kernel_sigma*kernel_gaussian
        )
        self.complexity_cost = _complexity_cost(
            self.kernel_sample,
            self.kernel_mean,
            self.kernel_sigma,
            self.kernel_prior_density
        )

        if self.use_bias:
            self.bias_mean = self.add_variable(
                'bias_mean',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
            self.bias_rho = self.add_variable(
                'bias_rho',
                shape=[self.units, ],
                initializer=self.bias_rho_initializer,
                dtype=self.dtype,
                trainable=True
            )
            self.bias_sigma = nn.softplus(self.bias_rho)
            bias_gaussian = random_ops.random_normal(
                name="bias_gaussian",
                shape=[self.units, ],
                dtype=self.dtype
                )
            self.bias_sample = (
                self.bias_mean
                + self.bias_sigma *bias_gaussian
            )
            self.complexity_cost += _complexity_cost(
                self.bias_sample,
                self.bias_mean,
                self.bias_sigma,
                self.bias_prior_density
            )
        else:
            self.bias_mean = None
            self.bias_rho = None
            self.bias_sigma = None
            self.bias_sample = None
        self.add_loss(self.complexity_cost)
        self.built = True

    def call(self, inputs, sample):
        kernel = control_flow_ops.cond(sample, lambda: self.kernel_sample, lambda: self.kernel_mean)

        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(
                inputs,
                kernel,
                [[len(shape) - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, kernel)
        if self.use_bias:
            bias = control_flow_ops.cond(sample, lambda: self.bias_sample, lambda: self.bias_mean)
            outputs = nn.bias_add(outputs, bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)
