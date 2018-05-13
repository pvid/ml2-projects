import tensorflow as tf


def _recurrent_step(c_prev, elems):
    f, i, z = tf.unstack(elems)
    return (f*c_prev) + (i*z)


def qrnn_layer(
    inputs, n_units, filter_width=2,
    pool_type='ifo', kernel_regularizer=None
):
    """
    Some very cool documentation
    """
    paddings = [[0, 0], [filter_width-1, 0], [0, 0]]
    inputs = tf.pad(inputs, paddings=paddings)
    inputs = tf.expand_dims(inputs, -1)
    input_dim = inputs.shape[2]

    # Candidate vectors
    candidate = tf.layers.conv2d(
        inputs,
        n_units,
        (filter_width, input_dim),
        activation=tf.tanh,
        kernel_regularizer=kernel_regularizer
    )
    candidate = tf.squeeze(candidate, axis=2)

    # Forget gate
    forget_gate = tf.layers.conv2d(
        inputs,
        n_units,
        (filter_width, input_dim),
        activation=tf.sigmoid,
        kernel_regularizer=kernel_regularizer
    )
    forget_gate = tf.squeeze(forget_gate, axis=2)

    # Input gate
    if 'i' in pool_type:
        input_gate = tf.layers.conv2d(
            inputs,
            n_units,
            (filter_width, input_dim),
            activation=tf.sigmoid
        )
        input_gate = tf.squeeze(input_gate, axis=2)
    else:
        input_gate = 1 - forget_gate

    # Recurrent part of the calculation of c
    # Prepare for tf.scan
    forget_gate = tf.transpose(forget_gate, [1, 0, 2])
    input_gate = tf.transpose(input_gate, [1, 0, 2])
    candidate = tf.transpose(candidate, [1, 0, 2])
    initializer = tf.zeros(tf.shape(forget_gate)[1:], tf.float32)

    # Apply reccurent step
    hidden = tf.scan(
        _recurrent_step,
        (forget_gate, input_gate, candidate),
        initializer=initializer
    )
    # Return to proper shape
    hidden = tf.transpose(hidden, [1, 0, 2])

    # Calculate outputs
    if 'o' in pool_type:
        # Output gate
        output_gate = tf.layers.conv2d(
            inputs,
            n_units,
            (filter_width, input_dim),
            activation=tf.sigmoid,
            kernel_regularizer=kernel_regularizer
        )
        output_gate = tf.squeeze(output_gate, axis=2)

        outputs = output_gate*hidden
    else:
        outputs = hidden

    return outputs
