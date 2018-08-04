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

CKPT_RES50 = './pretrained-checkpoint/resnet_v1_50.ckpt'
VALID_DIR = './origin-data/road-train-2+valid.v2/train_pick'
CKPT_TRAINED = './model/archive_0803/res50_u_net-21000'
CKPT_PARTIAL = False
# CKPT_TRAINED = './model/archive_0709/res50_u_net-87438'
# CKPT_PARTIAL = True

def test():
    settings = slim_model.Settings()
    with tf.Session() as sess:
        m_test = slim_model.Model()
        sess.run(tf.global_variables_initializer())
        if CKPT_PARTIAL:
            saver_pre_trained = tf.train.Saver(m_test.pretrained_variables)
            saver_pre_trained.restore(sess, CKPT_RES50)
            saver_trainable = tf.train.Saver(m_test.trainable_variables)
            saver_trainable.restore(sess, CKPT_TRAINED)
        else:
            saver_all = tf.train.Saver()
            saver_all.restore(sess, CKPT_TRAINED)
        print('restore complete')
        test_files = glob.glob(os.path.join(VALID_DIR, '*tif*'))
        valid_files = glob.glob(os.path.join(VALID_DIR, '*bmp*'))
        test_files = sorted(test_files)
        valid_files = sorted(valid_files)
        for i, file in enumerate(test_files):
            id = file[:file.rfind('_')]
            mask_file = '{}_out.jpg'.format(id)
            pred_file = '{}_pred.jpg'.format(id)
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
            imageio.imwrite(mask_file, np.squeeze(bin_pred))
            imageio.imwrite(pred_file, np.squeeze(pred))


if __name__ == '__main__':
    test()
