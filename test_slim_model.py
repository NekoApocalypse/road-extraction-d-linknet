import tensorflow as tf
import tensorflow.contrib.slim as slim
import slim_model
import data_loader
import numpy as np
import datetime
import time
import os
import glob
import imageio

<<<<<<< HEAD
CKPT_RES50 = './pretrained-checkpoint/resnet_v1_50.ckpt'
VALID_DIR = './origin-data/road-train-2+valid.v2/train_pick'
CKPT_TRAINED = './model/archive_0804/res50_u_net-9000'
CKPT_PARTIAL = False
# CKPT_TRAINED = './model/archive_0709/res50_u_net-87438'
# CKPT_PARTIAL = True

def test():
=======

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'valid_dir', './origin-data/road-train-2+valid.v2/valid',
    'path to valid files')
tf.app.flags.DEFINE_string(
    'ckpt_dir', './model/archive_0710',
    'path to saved model'
)


def main(_):
    valid_dir = FLAGS.valid_dir
    ckpt_dir = FLAGS.ckpt_dir
    test(valid_dir=valid_dir, ckpt_dir=ckpt_dir)


def test(valid_dir, ckpt_dir):
>>>>>>> 8cefab9d07ec7dfd9c8fd0deeb0414a41b213f1e
    settings = slim_model.Settings()

    checkpoint_state = tf.train.get_checkpoint_state(ckpt_dir)
    if not checkpoint_state:
        raise AssertionError('No valid checkpoints found in directory' +
                             '\'{}\''.format(ckpt_dir))
    input_checkpoint = checkpoint_state.model_checkpoint_path

    with tf.Session() as sess:
        m_test = slim_model.Model()
        sess.run(tf.global_variables_initializer())
        saver_all = tf.train.Saver()
        saver_all.restore(sess, input_checkpoint)
        print('restore complete')
        test_files = glob.glob(os.path.join(valid_dir, '*sat*'))
        valid_files = glob.glob(os.path.join(valid_dir, '*mask*'))
        test_files = sorted(test_files)
        valid_files = sorted(valid_files)
        print('{} files to test from {}'.format(len(test_files), valid_dir))
        for i, file in enumerate(test_files):
<<<<<<< HEAD
            id = file[:file.rfind('_')]
            id = id + '_' + str(i%50)
            mask_file = '{}_out.jpg'.format(id)
            pred_file = '{}_pred.jpg'.format(id)
=======
            file_id = file[:file.rfind('_')]
            mask_file = '{}_out.jpg'.format(file_id)
            pred_file = '{}_pred.jpg'.format(file_id)
>>>>>>> 8cefab9d07ec7dfd9c8fd0deeb0414a41b213f1e
            input_x = imageio.imread(file)
            input_x = input_x.astype(np.float32) / 255.0
            print('testing {}'.format(file))
            if valid_files:
                input_y = imageio.imread(valid_files[i])
                input_y = (input_y[:, :, 0] > 128).astype(np.int8)
                pred, bin_pred, loss, iou = sess.run(
                    [m_test.pred, m_test.bin_pred, m_test.dice_bce_loss, m_test.iou],
                    feed_dict={m_test.input_x: [input_x], m_test.input_y: [input_y]}
                )
                print('loss: {}, iou: {}'.format(loss, iou))
            else:
                pred, bin_pred= sess.run(
                    [m_test.pred, m_test.bin_pred],
                    feed_dict={m_test.input_x: [input_x]}
                )
<<<<<<< HEAD
            imageio.imwrite(mask_file, np.squeeze(bin_pred))
            imageio.imwrite(pred_file, np.squeeze(pred))
            print('save to {}'.format(mask_file))
            print('save to {}'.format(pred_file))
=======
            bin_pred = np.squeeze((bin_pred * 255.0).astype(np.uint8))
            pred = np.squeeze((pred * 255.0).astype(np.uint8))
            imageio.imwrite(mask_file, bin_pred)
            imageio.imwrite(pred_file, pred)
>>>>>>> 8cefab9d07ec7dfd9c8fd0deeb0414a41b213f1e


if __name__ == '__main__':
    tf.app.run()
