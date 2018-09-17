import tensorflow as tf
import tensorflow.contrib.slim as slim
import slim_model
import data_loader
import numpy as np
import datetime
import time
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary_dir', './summary', 'path to store summary')
tf.app.flags.DEFINE_string(
    'CKPT_RES50', './pretrained-checkpoint/resnet_v1_50.ckpt',
    'path to pre-trained res50 model')
tf.app.flags.DEFINE_string('save_dir', './model', 'path to save model')
tf.app.flags.DEFINE_string('resume_dir', '', 'path to resume training')
tf.app.flags.DEFINE_boolean(
    'no_append', False,
    'If set, model will be saved directly under `save_dir`. ' +
    'No sub directory will be made.'
)
tf.app.flags.DEFINE_string(
    'data_dir', './origin-data/road-train-2+valid.v2/train',
    'path to training data')
tf.app.flags.DEFINE_integer(
    'num_epoch', 16, 'number of epochs to train'
)
tf.app.flags.DEFINE_boolear(
    'partial_train', False,
    'If true, parameters in Res50 will not be updated.'
)


def assert_flags():
    assert os.path.isdir(FLAGS.summary_dir),\
        'Error, summary_dir must be a directory.'
    assert os.path.isfile(FLAGS.CKPT_RES50),\
        'Error, Res50 Model files not found. Please refer to readme.md.'
    assert os.path.isdir(FLAGS.save_dir),\
        'Error, save_dir must be a directory'
    if FLAGS.resume_dir:
        assert os.path.isdir(FLAGS.resume_dir),\
            'Error, resume_dir must be a directory'
    else:
        print('No resume_dir appointed, training from scratch.')
    assert os.path.isdir(FLAGS.data_dir),\
        'Error, data_dir must be a directory'


def main(_):
    assert_flags()
    summary_dir = FLAGS.summary_dir
    res50_dir = FLAGS.CKPT_RES50
    save_dir = FLAGS.save_dir
    no_append = FLAGS.no_append
    data_dir = FLAGS.data_dir
    resume_dir = FLAGS.resume_dir
    partial_train = FLAGS.partial_train
    train(
        data_dir=data_dir,
        resume_dir=resume_dir,
        save_dir=save_dir,
        res50_dir=res50_dir,
        summary_dir=summary_dir,
        no_append=no_append,
        partial_train=partial_train
    )


def train(data_dir, resume_dir, save_dir, res50_dir, summary_dir,
          no_append=False, partial_train=False):
    settings = slim_model.Settings()
    settings.num_epoch = FLAGS.num_epoch
    num_epoch = settings.num_epoch
    batch_size = settings.batch_size
    with tf.Session() as sess:
        m_train = slim_model.Model()
        # print(m_train.bin_pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=settings.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if not partial_train:
            train_op = optimizer.minimize(
                m_train.dice_bce_loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(
                m_train.dice_bce_loss,
                global_step=global_step,
                var_list=m_train.trainable_variables
            )
        sess.run(tf.global_variables_initializer())
        saver_pre_trained = tf.train.Saver(m_train.pretrained_variables)
        saver_pre_trained.restore(sess, res50_dir)
        # saver_trainable = tf.train.Saver(m_train.trainable_variables)
        saver_all = tf.train.Saver()
        if resume_dir:
            ckpt_state = tf.train.get_checkpoint_state(resume_dir)
            ckpt_path = ckpt_state.model_checkpoint_path
            saver_all.restore(sess, ckpt_path)
        print('restore complete')
        time_str = datetime.datetime.now().isoformat()
        time_str = time_str[:time_str.rfind('.')]
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(summary_dir, time_str), sess.graph
        )
        logger_path = os.path.join(summary_dir, 'log-{}.txt'.format(time_str))
        if no_append:
            saver_path = save_dir
        else:
            # create sub-directory under save_dir
            saver_path = os.path.join(save_dir, 'model_{}'.format(time_str))
        time_start = time.time()
        for epoch in range(num_epoch):
            data_gen = data_loader.ImageLoader(
                data_dir, buffer_size=200, shuffle=True)
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
                    time_str = time_str.replace(':', '-')
                    msg = '{}:step{}, loss: {:g}, iou: {:g}, dice_coeff: {:g}, x_sum: {:g}, x_bin_sum: {:g}, y_sum: {:g}'.format(
                        time_str, step, loss, iou, dice_coeff, x_sum, x_bin_sum, y_sum
                    )
                    print(msg)
                    with open(logger_path, 'a') as f:
                        print(msg, file=f)
                if step % 1000 == 0:
                    if not os.path.exists(saver_path):
                        os.makedirs(saver_path)
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
    tf.app.run()
