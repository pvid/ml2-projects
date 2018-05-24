import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

def _calculate_thresholds(means, sigmas, ps):
    concat_abs_mean = np.concatenate([np.abs(m).reshape(-1) for m in means])
    concat_sigma = np.concatenate([s.reshape(-1) for s in sigmas])
    abs_thres = np.percentile(concat_abs_mean, ps)
    snr_thres = np.percentile(concat_abs_mean/concat_sigma, ps)
    return abs_thres, snr_thres



def prune_eval(model, logdir):

    # Prepare data
    _, (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(10000, 28*28)
    x_test = x_test/126

    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(512)
    iterator = dataset.make_initializable_iterator()


    sample = tf.placeholder(tf.bool, shape=())
    x, labels = iterator.get_next()
    labels = tf.one_hot(labels, 10)

    logits = model(x, sample=sample)

    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_op = tf.metrics.accuracy(predictions, tf.argmax(labels, axis=1))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
        except (tf.errors.NotFoundError, ValueError) as e:
            print("Problem s loadovanim modelu")
        means_val = sess.run(model.means)
        sigmas_val = sess.run(model.sigmas)

    ps = [0, 10, 25, 50, 75, 90, 95, 98, 99]

    abs_thresholds, snr_thresholds = _calculate_thresholds(means_val, sigmas_val, ps)

    abs_ops = [model.absolute_value_prune(a) for a in abs_thresholds]
    snr_ops = [model.signal2noise_prune(a) for a in snr_thresholds]
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with tf.Session() as sess:
        print("Random pruning")
        for p in ps:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
            except (tf.errors.NotFoundError, ValueError) as e:
                print("Problem s loadovanim modelu")
            sess.run([iterator.initializer, tf.local_variables_initializer(), model.random_prune(p)])
            while True:
                try:
                    accuracy_v, progress_step = sess.run(
                        [accuracy_op,  logits], feed_dict={sample: False})

                except (tf.errors.OutOfRangeError, StopIteration):
                    print("Prune {}% test error: {:.2f}%"
                        .format(p, 100*(1-accuracy_v)))
                    break

        print("Absolute value pruning")
        try:
            saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
        except (tf.errors.NotFoundError, ValueError) as e:
            print("Problem s loadovanim modelu")

        for op, p in zip(abs_ops, ps):
            sess.run([iterator.initializer, tf.local_variables_initializer(), op])
            while True:
                try:
                    accuracy_v, progress_step = sess.run(
                        [accuracy_op,  logits], feed_dict={sample: False})

                except (tf.errors.OutOfRangeError, StopIteration):
                    print("Prune {}% test error: {:.2f}%"
                        .format(p, 100*(1-accuracy_v)))
                    break
        print("Signal-to-noise pruning")
        try:
            saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
        except (tf.errors.NotFoundError, ValueError) as e:
            print("Problem s loadovanim modelu")
        for op, p in zip(snr_ops , ps):
            sess.run([iterator.initializer, tf.local_variables_initializer(), op])
            while True:
                try:
                    accuracy_v, progress_step = sess.run(
                        [accuracy_op,  logits], feed_dict={sample: False})

                except (tf.errors.OutOfRangeError, StopIteration):
                    print("Prune {}% test error: {:.2f}%"
                        .format(p, 100*(1-accuracy_v)))
                    break
