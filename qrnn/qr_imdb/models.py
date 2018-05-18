import tensorflow as tf
from keras.layers import LSTM, CuDNNLSTM, Dropout, Dense

from .qrnn import qr_layer
from .qrnn_alternative import QRLayer


def model_qrnn(x, is_training, embed_weights):
    filter_width = 2
    pool_type = 'ifo'
    n_units = 256
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=10**(-6))
    with tf.variable_scope('QRNN', reuse=tf.AUTO_REUSE):

        # GloVe embedding
        W = tf.get_variable(
            name="W",
            shape=embed_weights.shape,
            initializer=tf.constant_initializer(embed_weights),
            trainable=False)

        embedded = tf.nn.embedding_lookup(
            W, x, name="glove_embedding")
        embedded = tf.layers.dropout(
            embedded,
            rate=0.5,
            training=is_training)

        outputs = qr_layer(
            embedded,
            n_units,
            filter_width,
            pool_type,
            kernel_regularizer
        )

        stacked = tf.concat([embedded, outputs], axis=2)
        stacked = tf.layers.dropout(
            stacked,
            rate=0.5,
            training=is_training)

        outputs = qr_layer(
            stacked,
            n_units,
            filter_width,
            pool_type,
            kernel_regularizer
        )
        outputs = outputs[:, -1, :]
        outputs = tf.layers.dropout(
            outputs,
            rate=0.5,
            training=is_training)

        out = tf.layers.dense(outputs, 1)

    return tf.squeeze(out)

def model_qrnn_alt(x, is_training, embed_weights):
    filter_width = 2
    pool_type = 'ifo'
    n_units = 256
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=10**(-6))
    with tf.variable_scope('QRNN', reuse=tf.AUTO_REUSE):

        # GloVe embedding
        W = tf.get_variable(
            name="W",
            shape=embed_weights.shape,
            initializer=tf.constant_initializer(embed_weights),
            trainable=False)

        embedded = tf.nn.embedding_lookup(
            W, x, name="glove_embedding")
        embedded = tf.layers.dropout(
            embedded,
            rate=0.5,
            training=is_training)

        qr1 = QRLayer(
            n_units,
            filter_width,
            pool_type,
            kernel_regularizer
        )
        outputs = qr1(embedded)

        stacked = tf.concat([embedded, outputs], axis=2)
        stacked = tf.layers.dropout(
            stacked,
            rate=0.5,
            training=is_training)

        qr2 = QRLayer(
            n_units,
            filter_width,
            pool_type,
            kernel_regularizer
        )

        outputs = qr2(stacked)

        outputs = outputs[:, -1, :]
        outputs = tf.layers.dropout(
            outputs,
            rate=0.5,
            training=is_training)

        out = tf.layers.dense(outputs, 1)

    return tf.squeeze(out)


def model_cudnnlstm(x, is_training, embed_weights):
    with tf.variable_scope('LSTMNet', reuse=tf.AUTO_REUSE):
        # GloVe embedding
        W = tf.get_variable(
            name="W",
            shape=embed_weights.shape,
            initializer=tf.constant_initializer(embed_weights),
            trainable=False)
        embedded = tf.nn.embedding_lookup(
            W, x, name="glove_embedding")
        embedded = tf.layers.dropout(
            embedded,
            rate=0.5,
            training=is_training)

        outputs = CuDNNLSTM(256, return_sequences=True)(embedded)

        stacked = tf.concat([embedded, outputs], axis=2)
        stacked = Dropout(0.5)(stacked)

        outputs = CuDNNLSTM(256, return_sequences=False)(stacked)
        outputs = Dropout(0.5)(outputs)

        out = Dense(1)(outputs)

    return tf.squeeze(out)

def model_lstm(x, is_training, embed_weights):
    with tf.variable_scope('LSTMNet', reuse=tf.AUTO_REUSE):
        # GloVe embedding
        W = tf.get_variable(
            name="W",
            shape=embed_weights.shape,
            initializer=tf.constant_initializer(embed_weights),
            trainable=False)
        embedded = tf.nn.embedding_lookup(
            W, x, name="glove_embedding")
        embedded = tf.layers.dropout(
            embedded,
            rate=0.5,
            training=is_training)

        outputs = LSTM(256, return_sequences=True)(embedded)

        stacked = tf.concat([embedded, outputs], axis=2)
        stacked = Dropout(0.5)(stacked)

        outputs = LSTM(256, return_sequences=False)(stacked)
        outputs = Dropout(0.5)(outputs)

        out = Dense(1)(outputs)

    return tf.squeeze(out)
