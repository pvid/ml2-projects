import tensorflow as tf


class QRNNLayer:

    def __init__(self, n_units, filter_width, pool_type, kernel_regularizer=None):
        self._n_units = n_units
        self._filter_width = filter_width
        if pool_type not in ['f', 'fo', 'ifo']:
            raise ValueError("Pool type must be one of 'f', 'fo', 'ifo'")
        self._pool_type = pool_type
        self._kernel_regularizer = kernel_regularizer

    def __call__(self, inputs):
        paddings = [[0, 0], [self._filter_width-1, 0], [0, 0]]
        inputs = tf.pad(inputs, paddings=paddings)
        inputs = tf.expand_dims(inputs, -1)
        input_dim = inputs.get_shape()[2]

        z = tf.layers.conv2d(
            inputs,
            self._n_units,
            (self._filter_width, input_dim),
            activation=tf.tanh,
            kernel_regularizer=self._kernel_regularizer
        )
        z = tf.squeeze(z, axis=2)

        conv_units = (len(self._pool_type))*self._n_units

        gate_conv = tf.layers.conv2d(
            inputs,
            conv_units,
            (self._filter_width, input_dim),
            activation=tf.sigmoid,
            kernel_regularizer=self._kernel_regularizer
        )
        gate_conv = tf.squeeze(gate_conv, axis=2)

        pool = QRNNPool(self._n_units, self._pool_type)

        c = tf.concat([z, gate_conv], axis=2)

        outputs, _ = tf.nn.dynamic_rnn(
            cell=pool,
            inputs=tf.concat([z, gate_conv], axis=2),
            dtype=tf.float32
        )

        return outputs


class QRNNPool(tf.nn.rnn_cell.RNNCell):

    def __init__(self, n_units, pool_type):
        self._n_units = n_units
        if pool_type not in ['f', 'fo', 'ifo']:
            raise ValueError("Pool type must be one of 'f', 'fo', 'ifo'")
        self._pool_type = pool_type

    @property
    def state_size(self):
        return self._n_units

    @property
    def output_size(self):
        return self._n_units

    def __call__(self, inputs, state):
        if self._pool_type == 'f':
            z, f = tf.split(inputs, 2, 1)
            output = f*state + (1-f)*z
            next_state = output
        elif self._pool_type == 'fo':
            z, f, o = tf.split(inputs, 3, 1)
            next_state = f*state + (1-f)*z
            output = o*next_state
        elif self._pool_type == 'ifo':
            z, f, o, i = tf.split(inputs, 4, 1)
            next_state = f*state + i*z
            output = o*next_state
        return output, next_state
