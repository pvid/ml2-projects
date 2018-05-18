from tqdm import tqdm
import tensorflow as tf

from .imdb_datasets import imdb_datasets


def train_eval_loop(
    epochs, embed_weights, model, logdir='./model'):

    # Prepare datasets
    train, validation = imdb_datasets(150)

    iterator: tf.data.Iterator = (
        tf.data.Iterator.from_structure(
            train.output_types, train.output_shapes
            )
    )

    # Model, loss, summaries
    x, labels = iterator.get_next()
    training = tf.placeholder(tf.bool, shape=())
    global_step = tf.Variable(
        0, name='global_step', dtype=tf.int32, trainable=False
    )

    logits = model(x, training, embed_weights)

    loss = tf.losses.sigmoid_cross_entropy(labels, logits, scope='loss')
    loss += tf.losses.get_regularization_loss()
    train_op = (
        tf.train.AdamOptimizer()
        .minimize(loss, global_step=global_step)
    )

    predictions = tf.cast(logits > 0, tf.int32)
    accuracy, accuracy_op = tf.metrics.accuracy(predictions, labels)
    mean_loss, mean_loss_op = tf.metrics.mean(loss)

    loss_summary = tf.summary.scalar('loss', mean_loss)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge([loss_summary, accuracy_summary])

    train_writer = tf.summary.FileWriter(logdir+"/train")
    validation_writer = tf.summary.FileWriter(logdir+"/validation")

    saver = tf.train.Saver()

    # Train loop

    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(logdir+'/saves'))
        except (tf.errors.NotFoundError, ValueError) as e:
            sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            sess.run(iterator.make_initializer(train))

            sess.run(tf.local_variables_initializer())

            tqdm.write(f'Training on epoch {epoch}')
            bar = tqdm(total=25000)
            while True:
                try:

                    _, step, _, _, progress_step = (
                        sess.run(
                            [train_op, global_step, accuracy_op,
                             mean_loss_op, logits],
                            feed_dict={training: True}
                            )
                    )
                    progress_step = progress_step.shape[0]
                    bar.update(progress_step)

                    accuracy_v, mean_loss_v = sess.run([accuracy, mean_loss])
                    if step % 50 == 0:
                        summary = sess.run(merged_summary)
                        train_writer.add_summary(summary, step)
                        train_writer.flush()
                        sess.run(tf.local_variables_initializer())

                except (tf.errors.OutOfRangeError, StopIteration):
                    bar.close()
                    break

            sess.run(iterator.make_initializer(validation))

            sess.run(tf.local_variables_initializer())

            tqdm.write(f'Evaluating on epoch {epoch}')
            bar = tqdm(total=25000)
            while True:
                try:
                    accuracy_v, mean_loss_v, progress_step = (
                        sess.run(
                            [accuracy_op, mean_loss_op, logits],
                            feed_dict={training: False}
                        )
                    )
                    progress_step = progress_step.shape[0]
                    bar.update(progress_step)

                except (tf.errors.OutOfRangeError, StopIteration):
                    summary = sess.run(merged_summary)
                    validation_writer.add_summary(summary, step)
                    validation_writer.flush()
                    bar.set_postfix(
                        mean_loss=mean_loss_v,
                        accuracy=accuracy_v
                    )
                    bar.close()
                    break

            tqdm.write('------')
            saver.save(sess, logdir+'/saves/save', global_step=step)
