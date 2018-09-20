import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import vgg
from nets import resnet_v1
import numpy as np


CKPT_RES50 = './pretrained-checkpoint/resnet_v1_50.ckpt'
CKPT_VGG16 = './pretrained-checkpoint/vgg_16.ckpt'


class Settings(object):
    def __init__(self):
        self.dim_x = 1024
        self.dim_y = 1024
        self.pretrained_model = 'RES50'
        self.learning_rate = 0.0001
        self.num_epoch = 16
        self.batch_size = 2
        self.threshold = 0.5
        self.dice_smooth = 1
        self.bce_dice_weights = (0.5, 0.5)
        self.l2_weight = 0.0001


def print_endpoints(endpoints, file_name):
    with open(file_name, 'w') as f:
        for name, op in endpoints.items():
            print(name, op.shape.as_list(), file=f)


class Model(object):
    def __init__(self):
        self.settings = Settings()
        self.build_net()

    def build_net(self):
        def up_conv_block(input, output_dim, base_dim):
            block = slim.conv2d(input, base_dim, [1, 1])
            block = slim.convolution2d_transpose(block, base_dim, [3, 3], stride=2)
            block = slim.conv2d(block, output_dim, [1, 1])
            return block

        settings = Settings()
        dim_x, dim_y = settings.dim_x, settings.dim_y
        self.input_x = tf.placeholder(
            tf.float32, [None, dim_x, dim_y, 3], 'input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, dim_x, dim_y], 'input_y')
        input_y_exp = tf.expand_dims(self.input_y, -1)
        if settings.pretrained_model == 'VGG16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                net, endpoints = vgg.vgg_16(
                    self.input_x, global_pool=False, is_training=False,
                    spatial_squeeze=False)
        elif settings.pretrained_model == 'RES50':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, endpoints = resnet_v1.resnet_v1_50(
                    self.input_x, global_pool=False, is_training=False)
        else:
            raise ValueError('pretrained model type {} not recognised.'.format(
                settings.pretrained_model))
        pretrained_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        '''
        with tf.variable_scope('dummy_out'):
            dummy_loss = tf.reduce_mean(net)
            print_endpoints(endpoints, 'endpoints.txt')
        '''
        with tf.variable_scope('bridge'):
            bridge_output = []
            net = slim.conv2d(net, 512, [3, 3])
            bridge_output.append(net)
            net = tf.layers.conv2d(net, 512, [3, 3], dilation_rate=(2, 2),
                                   activation=tf.nn.relu, padding='same')
            bridge_output.append(net)
            net = tf.layers.conv2d(net, 512, [3, 3], dilation_rate=(4, 4),
                                   activation=tf.nn.relu, padding='same')
            bridge_output.append(net)
            net = tf.layers.conv2d(net, 512, [3, 3], dilation_rate=(8, 8),
                                   activation=tf.nn.relu, padding='same')
            bridge_output.append(net)
            net = tf.add_n(bridge_output)
        # print('bridge output')
        # print(net)
        with tf.variable_scope('decoder'):
            net = up_conv_block(net, 1024, 256)
            bridged = endpoints['resnet_v1_50/block3/unit_5/bottleneck_v1']
            net = net + bridged
            # 64, 64, 1024
            net = up_conv_block(net, 512, 128)
            bridged = endpoints['resnet_v1_50/block2/unit_3/bottleneck_v1']
            net = net + bridged
            # 128, 128, 512
            net = up_conv_block(net, 256, 64)
            bridged = endpoints['resnet_v1_50/block1/unit_2/bottleneck_v1']
            net = net + bridged
            # 256, 256, 256
            net = up_conv_block(net, 64, 16)
            # 512, 512, 64
            net = slim.conv2d_transpose(net, 32, [4, 4], stride=2)
            # 1024, 1024, 32
            net = slim.conv2d(net, 1, [3, 3], activation_fn=None)
        with tf.variable_scope('metrics'):
            self.output = net
            self.pred = tf.nn.sigmoid(net)
            self.bce_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=input_y_exp, logits=self.output
                )
            )
            pred_flat = tf.layers.flatten(self.pred)
            y_flat = tf.layers.flatten(input_y_exp)
            self.dice_coeff = 2 * tf.reduce_sum(pred_flat * y_flat) / \
                (tf.reduce_sum(pred_flat) + tf.reduce_sum(y_flat))
            self.dice_loss = 1 - self.dice_coeff
            w1, w2 = settings.bce_dice_weights
            self.dice_bce_loss = w1 * self.bce_loss + w2 * self.dice_loss
            self.bin_pred = tf.cast(self.pred > settings.threshold, tf.int32)
            input_y_bin = tf.cast(input_y_exp > settings.threshold, tf.int32)
            self.debug_x_sum = tf.reduce_sum(self.pred)
            self.debug_x_bin_sum = tf.reduce_sum(self.bin_pred)
            self.debug_y_sum = tf.reduce_sum(input_y_exp)
            self.iou = tf.reduce_sum(self.bin_pred * input_y_bin) / \
                (tf.reduce_sum(
                    tf.cast((self.bin_pred + input_y_bin) > 0, tf.int32)
                ))
            tf.summary.scalar('bce_loss', self.bce_loss)
            tf.summary.scalar('dice_coeff', self.dice_coeff)
            tf.summary.scalar('bce_dice_loss', self.dice_bce_loss)
            # Regularization
            self.l2_loss = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(settings.l2_weight),
                weights_list=tf.trainable_variables()
            )
            self.dice_bce_l2_loss = self.dice_bce_loss + self.l2_loss
            tf.summary.scalar('l2_loss', self.l2_loss)
            tf.summary.scalar('dice_bce_l2_loss', self.dice_bce_l2_loss)

        self.pretrained_variables = pretrained_variables
        self.pretrained_endpoints = endpoints
        self.trainable_variables = []
        self.trainable_variables.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bridge')
        )
        self.trainable_variables.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        )


def unit_test():
    settings = Settings()
    with tf.Session() as sess:
        model = Model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model.pretrained_variables)
        if settings.pretrained_model == 'VGG16':
            saver.restore(sess, CKPT_VGG16)
        elif settings.pretrained_model == 'RES50':
            saver.restore(sess, CKPT_RES50)
        print('restore complete')
        print(model.output)
        print(model.trainable_variables)
        print(model.bce_loss)
        print(model.dice_bce_l2_loss)


def collect_output_nodes():
    settings = Settings()
    with tf.Session() as sess:
        model = Model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model.pretrained_variables)
        if settings.pretrained_model == 'VGG16':
            saver.restore(sess, CKPT_VGG16)
        elif settings.pretrained_model == 'RES50':
            saver.restore(sess, CKPT_RES50)
        print('restore complete')
    output_nodes = [
        model.pred,
        model.bin_pred
    ]
    print(output_nodes)


if __name__ == '__main__':
    unit_test()
    collect_output_nodes()

