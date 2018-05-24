from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist


def train_loop(model, logdir, epochs=20):

    batch_size = 128
    learning_rate = 0.001

    # Prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28*28)
    x_train = x_train/126

    x_test = x_test.reshape(10000, 28*28)
    x_test = x_test/126

    features_placeholder = tf.placeholder(x_train.dtype, [None, 28*28])
    labels_placeholder = tf.placeholder(y_train.dtype, [None, ])

    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    dataset = dataset.batch(batch_size).shuffle(5000)
    iterator = dataset.make_initializable_iterator()

    global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
    sample = tf.placeholder(tf.bool, shape=())

    x, labels = iterator.get_next()
    labels = tf.one_hot(labels, 10)

    logits = model(x, sample=sample)

    predictions = tf.argmax(logits, axis=1)

    loss = tf.losses.softmax_cross_entropy(labels, logits, scope='loss')


    # add complexity cost
    complexity_cost = model.complexity_cost
    loss_with_complexity = loss+(complexity_cost/60000)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_with_complexity, global_step=global_step)

    # Setup validation metrics
    accuracy, accuracy_op = tf.metrics.accuracy(predictions, tf.argmax(labels, axis=1))
    mean_loss, mean_loss_op = tf.metrics.mean(loss)
    mean_loss_with_complexity, mean_loss_with_complexity_op = tf.metrics.mean(loss_with_complexity)

    loss_summary = tf.summary.scalar('loss', mean_loss)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    loss_with_complexity_summary = tf.summary.scalar('loss_with_complexity', mean_loss_with_complexity)

    merged_summary = tf.summary.merge([loss_summary, accuracy_summary, loss_with_complexity_summary])

    train_writer = tf.summary.FileWriter(logdir+"/train")
    validation_writer = tf.summary.FileWriter(logdir+"/validation")

    saver = tf.train.Saver()

    from collections import OrderedDict

    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
        except (tf.errors.NotFoundError, ValueError) as e:
            # Log that model was not found
            print("Model not found")
            sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            sess.run(iterator.initializer, feed_dict={features_placeholder: x_train,
                                              labels_placeholder: y_train})
            # zero-out metrics, they're kept in the local variables collection
            sess.run(tf.local_variables_initializer())

            # start a progress bar and iterate until dataset exhaustion
            tqdm.write(f'Training on epoch {epoch}')
            bar = tqdm(total=60000)
            while True:
                try:
                    # run train iteration
                    _, _, step, _, _, progress_step  = sess.run(
                        [train_op, mean_loss_with_complexity_op, global_step, accuracy_op, mean_loss_op, logits],
                        feed_dict={sample: True})


                    # update progress bar with batch size
                    progress_step = progress_step.shape[0]
                    bar.update(progress_step)

                    accuracy_v, mean_loss_v = sess.run([accuracy, mean_loss])
                    if step % 50 == 0:
                        postfix = OrderedDict(loss=f'{accuracy_v:.4f}', accuracy=f'{mean_loss_v:.4f}')
                        summary = sess.run(merged_summary)
                        train_writer.add_summary(summary, step)
                        train_writer.flush()
                        sess.run(tf.local_variables_initializer())

                except (tf.errors.OutOfRangeError, StopIteration):
                    bar.close()
                    break

            # initialize iterator for validation dataset. No need to shuffle
            sess.run(iterator.initializer, feed_dict={features_placeholder: x_test,
                                              labels_placeholder: y_test})

            # zero-out metrics for evaluation
            sess.run(tf.local_variables_initializer())

            tqdm.write(f'Evaluating on epoch {epoch}')
            bar = tqdm(total=10000)
            while True:
                try:
                    accuracy_v, mean_loss_v, progress_step, _ = sess.run(
                        [accuracy_op, mean_loss_op, logits, mean_loss_with_complexity_op], feed_dict={sample: False})


                    progress_step = progress_step.shape[0]
                    bar.update(progress_step)
                except (tf.errors.OutOfRangeError, StopIteration):
                    summary = sess.run(merged_summary)
                    validation_writer.add_summary(summary, step)
                    validation_writer.flush()
                    postfix = OrderedDict(loss=f'{mean_loss_v:.4f}', accuracy=f'{accuracy_v:.4f}')
                    bar.set_postfix(postfix)
                    bar.close()
                    break

            tqdm.write('------')
            saver.save(sess, logdir+"/saves/save", global_step=step)
