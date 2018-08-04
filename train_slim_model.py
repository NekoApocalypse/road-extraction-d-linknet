import tensorflow as tf
import tensorflow.contrib.slim as slim
import slim_model
import data_loader
import numpy as np
import datetime
import time
import os


CKPT_RES50 = './pretrained-checkpoint/resnet_v1_50.ckpt'
SAVE_DIR = './model'
SUMMARY_DIR = './summary'


def train(resume_dir=None):
    """ Train the slim model.
    Args:
        resume_path (string): If set, resumes training from checkpoint file.
    """
    settings = slim_model.Settings()
    num_epoch = settings.num_epoch
    batch_size = settings.batch_size
    with tf.Session() as sess:
        m_train = slim_model.Model()
        optimizer = tf.train.AdamOptimizer(learning_rate=settings.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(
            m_train.dice_bce_loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        saver_pre_trained = tf.train.Saver(m_train.pretrained_variables)
        saver_pre_trained.restore(sess, CKPT_RES50)
        # saver_trainable = tf.train.Saver(m_train.trainable_variables)
        saver_all = tf.train.Saver()
        if resume_dir is not None:
            saver_all.restore(sess, resume_dir)
        print('restore complete')
        time_str = datetime.datetime.now().isoformat()
        time_str = time_str[:time_str.rfind('.')]
        time_str = time_str.replace(':', '_')
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(SUMMARY_DIR, time_str), sess.graph
        )
        logger_path = os.path.join(SUMMARY_DIR, 'log-{}.txt'.format(time_str))
        saver_path = os.path.join(SAVE_DIR, 'model_{}'.format(time_str))
        os.makedirs(saver_path)
        time_start = time.time()
        for epoch in range(num_epoch):
            # data loader
            data_gen = data_loader.ImageLoader(buffer_size=200, shuffle=True, num_slices=2)
            while not data_gen.finished():
                inputs_x, inputs_y = data_gen.serve_data(batch_size)
                feed_dict = {
                    m_train.input_x: inputs_x,
                    m_train.input_y: inputs_y
                }
                _, step, loss, iou, dice_coeff, x_sum, x_bin_sum, y_sum, summary = sess.run(
                    [train_op, global_step, m_train.dice_bce_loss,
                     m_train.iou, m_train.dice_coeff, m_train.debug_x_sum,
                     m_train.debug_x_bin_sum, m_train.debug_y_sum,
                     merged_summary],
                    feed_dict=feed_dict
                )
                summary_writer.add_summary(summary, step)
                if step % 10 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    time_str = time_str[:time_str.rfind('.')]
                    msg = '{}:step{}, loss: {:g}, iou: {:g}, dice_coeff: {:g}, x_sum: {:g}, x_bin_sum: {:g}, y_sum: {:g}'.format(
                        time_str, step, loss, iou, dice_coeff, x_sum, x_bin_sum, y_sum
                    )
                    print(msg)
                    with open(logger_path, 'a') as f:
                        print(msg, file=f)
                if step % 1000 == 0:
                    path = saver_all.save(
                        sess, os.path.join(saver_path, 'res50_u_net'),
                        global_step=global_step
                    )
                    print('Saved model to {}.'.format(path))
        time_used = time.time() - time_start
        print('Training Finished. Time used: {}'.format(
            datetime.timedelta(seconds=time_used)
        ))
        print('Saving model...')
        path = saver_all.save(
            sess, os.path.join(saver_path, 'res50_u_net'),
            global_step=global_step
        )
        print('Saved model to {}.'.format(path))


if __name__ == '__main__':
    train(resume_dir=None)
