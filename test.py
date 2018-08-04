import numpy as np
import os
import glob
import tensorflow as tf
import network
import imageio
import data_loader
import datetime
import time

SAVE_DIR = './model'
test_dir = './origin-data/road-train-2+valid.v2/valid'
view_dir = './origin-data/preview'


def real2grey(img):
    settings = network.Settings()
    img_size = settings.input_size[:2]
    img_map = np.reshape(img, img_size)
    img_map = (img_map * 255).astype(np.uint8)
    return img_map


def visualize_test(model, file_list, num_slice, sess):
    for file in file_list:
        img_large = imageio.imread(file)
        v_slices = np.split(img_large, num_slice)
        frags = [
            frag for v_slice in v_slices
            for frag in np.split(v_slice, num_slice, axis=1)
        ]
        for idx, frag in enumerate(frags):
            img_file = os.path.basename(file)
            img_file = os.path.splitext(img_file)[0]
            img_file = os.path.join(view_dir, img_file)
            slice_file = '{}-slice-{}.png'.format(img_file, idx)
            logits_file = '{}-slice-{}-logits.png'.format(img_file, idx)
            mask_file = '{}-slice-{}-mask.png'.format(img_file, idx)
            feed_dict = {model.inputs_x: [frag]}
            [score, pred] = sess.run([model.score, model.pred],
                                     feed_dict=feed_dict)
            score_img = real2grey(score)
            pred_img = real2grey(pred)
            imageio.imwrite(slice_file, frag)
            imageio.imwrite(logits_file, score_img)
            # imageio.imwrite(mask_file, pred_img)


def test():
    settings = network.Settings()
    with tf.Session() as sess:
        with tf.variable_scope('model'):
            m_test = network.SimpleNet(settings, is_training=False)
        saver = tf.train.Saver()
        print('Restoring model...')
        saver.restore(sess, os.path.join(SAVE_DIR, 'res-u-net-44840'))
        print('Restore complete.')
        file_list = glob.glob(
            os.path.join(test_dir, '*sat*')
        )
        file_list = sorted(file_list)
        file_list_small = file_list[:1]
        visualize_test(m_test, file_list_small, settings.num_slices, sess)


if __name__ == '__main__':
    test()

