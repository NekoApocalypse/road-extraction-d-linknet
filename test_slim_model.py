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
from elephant_in_the_freezer import load_graph


# Force test to run on cpu only
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'input_dir', '',
    'path to input files')
tf.app.flags.DEFINE_string(
    'output_dir', '',
    'path to output files'
)
tf.app.flags.DEFINE_string(
    'ckpt_dir', '',
    'path to saved model'
)
tf.app.flags.DEFINE_string(
    'pb_dir', '',
    'path to frozen model. If set, overrides ckpt_dir.'
)


def main(_):
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    ckpt_dir = FLAGS.ckpt_dir
    pb_dir = FLAGS.pb_dir
    # path assertion
    assert os.path.isdir(input_dir),\
        'Error, input_dir must be a directory'
    if output_dir:
        assert os.path.isdir(output_dir),\
            'Error, output_dir must be a directory'
    if pb_dir:
        assert os.path.isdir(pb_dir),\
            'Error, pb_dir must be a directory'
    elif ckpt_dir:
        assert os.path.isdir(ckpt_dir),\
            'Error, ckpt_dir must be a directory'
    else:
        raise(AssertionError('Error, either ckpt_dir or pb_dir must be set'))

    if not output_dir:
        output_dir = input_dir
    if pb_dir:
        test_from_frozen_graph(
            input_dir=input_dir,
            output_dir=output_dir,
            pb_dir=pb_dir
        )
    else:
        test(input_dir=input_dir,
             output_dir=output_dir,
             ckpt_dir=ckpt_dir
        )


def test_from_frozen_graph(input_dir, output_dir, pb_dir):
    print('Testing from fronzen model')
    graph, endpoints = load_graph(pb_dir)
    x_tensor = endpoints['x']
    pred_tensor = endpoints['pred']
    bin_pred_tensor = endpoints['bin_pred']
    with tf.Session(graph=graph) as sess:
        test_files = glob.glob(os.path.join(input_dir, '*sat*'))
        test_files = sorted(test_files)
        for i, file in enumerate(test_files):
            file_name = os.path.basename(file)
            file_id = file_name[:file_name.rfind('_')]
            mask_file = '{}_out.bmp'.format(os.path.join(output_dir, file_id))
            pred_file = '{}_pred.bmp'.format(os.path.join(output_dir, file_id))
            input_x = imageio.imread(file)
            input_x = input_x.astype(np.float32) / 255.0
            print('testing {}'.format(file))
            pred, bin_pred = sess.run(
                [pred_tensor, bin_pred_tensor],
                feed_dict={x_tensor: [input_x]}
            )
            bin_pred = np.squeeze((bin_pred * 255.0).astype(np.uint8))
            pred = np.squeeze((pred * 255.0).astype(np.uint8))
            print('    Saving to {}'.format(mask_file))
            imageio.imwrite(mask_file, bin_pred)
            print('    Saving to {}'.format(pred_file))
            imageio.imwrite(pred_file, pred)


def test(input_dir, output_dir, ckpt_dir):
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
        test_files = glob.glob(os.path.join(input_dir, '*sat*'))
        valid_files = glob.glob(os.path.join(input_dir, '*mask*'))
        test_files = sorted(test_files)
        valid_files = sorted(valid_files)
        print('{} files to test from {}'.format(len(test_files), input_dir))
        for i, file in enumerate(test_files):
            file_name = os.path.basename(file)
            file_id = file_name[:file_name.rfind('_')]
            mask_file = '{}_out.bmp'.format(os.path.join(output_dir, file_id))
            pred_file = '{}_pred.bmp'.format(os.path.join(output_dir, file_id))
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
            bin_pred = np.squeeze((bin_pred * 255.0).astype(np.uint8))
            pred = np.squeeze((pred * 255.0).astype(np.uint8))
            imageio.imwrite(mask_file, bin_pred)
            imageio.imwrite(pred_file, pred)


if __name__ == '__main__':
    tf.app.run()
