import numpy as np
import os
import tensorflow as tf
import network
import data_loader
import datetime
import time

SUMMARY_DIR = './summary'
SAVE_DIR = './model'


def train():
    settings = network.Settings()
    num_epoch = settings.num_epoch
    batch_size = settings.batch_size
    learning_rate = settings.learning_rate
    num_slices = settings.num_slices

    with tf.Session() as sess:
        with tf.variable_scope('model'):
            m_train = network.SimpleNet(settings, is_training=True)
        global_step = tf.Variable(0, name='Global_Step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(m_train.loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        time_start = time.time()
        time_str = datetime.datetime.now().isoformat()
        time_str = time_str[:time_str.rfind('.')]
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(SUMMARY_DIR, time_str), sess.graph
        )
        for epoch in range(num_epoch):
            data_gen = data_loader.ImageLoader(
                buffer_size=200, num_slices=num_slices)
            while not data_gen.finished():
                inputs_x, inputs_y = data_gen.serve_data(batch_size)
                feed_dict = {
                    m_train.inputs_x: inputs_x,
                    m_train.inputs_y: inputs_y
                }
                _, step, loss, accuracy, summary = sess.run(
                    [train_op, global_step, m_train.loss,
                     m_train.accuracy, merged_summary],
                    feed_dict=feed_dict
                )
                summary_writer.add_summary(summary, step)

                if step % 10 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    time_str = time_str[:time_str.rfind('.')]
                    tmp_str = '{}: step{}, loss: {:g}, acc: {:g}'.format(
                        time_str, step, loss, accuracy
                    )
                    print(tmp_str)

        time_used = time.time() - time_start
        print('Training Finished. Time used: {}'.format(
            datetime.timedelta(seconds=time_used)))
        print('Saving model...')
        path = saver.save(sess,
                          os.path.join(SAVE_DIR, 'res-u-net'),
                          global_step=step)
        tmp_str = 'Saved model to {}'.format(path)
        print(tmp_str)


if __name__ == '__main__':
    train()
