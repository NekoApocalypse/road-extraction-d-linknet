import tensorflow as tf

slim = tf.contrib.slim


class Settings(object):
    def __init__(self):
        self.num_epoch = 5
        self.batch_size = 4
        self.learning_rate = 0.001
        self.num_slices = 4
        resolution = 1024 // self.num_slices
        self.input_size = (resolution, resolution, 3)


class SimpleNet(object):
    def __init__(self, settings, is_training=False):
        self.settings = settings
        (input_rows, input_cols, input_channels) = settings.input_size
        self.inputs_x = tf.placeholder(
            tf.float32, [None, input_rows, input_cols, input_channels],
            name='inputs-x'
        )
        self.inputs_y = tf.placeholder(
            tf.int32, [None, input_rows, input_cols],
            name='inputs-y'
        )
        w1 = tf.get_variable('weight-1', [5, 5, input_channels, 32])
        c1 = tf.nn.conv2d(
            self.inputs_x, w1, strides=[1, 1, 1, 1], padding='SAME')
        c1_act = tf.nn.relu(c1)
        w2 = tf.get_variable('weight-2', [3, 3, 32, 64])
        c2 = tf.nn.conv2d(
            c1_act, w2, strides=[1, 1, 1, 1], padding='SAME'
        )
        c2_act = tf.nn.relu(c2)
        w3 = tf.get_variable('weight-3', [3, 3, 64, 32])
        c3 = tf.nn.conv2d(
            c2_act, w3, strides=[1, 1, 1, 1], padding='SAME'
        )
        c3_act = tf.nn.relu(c3)
        w4 = tf.get_variable('weight-4', [5, 5, 32, 1])
        c4 = tf.nn.conv2d(
            c3_act, w4, strides=[1, 1, 1, 1], padding='SAME'
        )

        logits = tf.reshape(c4, [-1, input_rows * input_cols])
        self.score = tf.nn.sigmoid(logits)
        self.pred = tf.cast(
            tf.greater_equal(tf.nn.sigmoid(logits), 0.5), tf.int32
        )
        labels = tf.reshape(self.inputs_y, [-1, input_rows * input_cols])
        diff = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32),
            logits=logits
        )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(diff)
        with tf.variable_scope('acc'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(labels, self.pred), tf.float32)
            )
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)


class ResUNet(object):
    def multi_res_block(self, x, num, filters, kernel_size, is_training=False):
        net = x
        for _ in num:
            net = self.res_block(net, filters, kernel_size, is_training)
        return net

    def res_block(self, x, filters, kernel_size, is_training=False):
        conv = slim.repeat(
            x, 2, slim.conv2d, filters, kernel_size, padding='same')
        return conv + x

    def __init__(self, settings, is_training=False):
        self.settings = settings
        (input_rows, input_cols, input_channels) = settings.input_size
        # build network

        self.inputs_x = tf.placeholder(
            tf.float32, [None, input_rows, input_cols, input_channels],
            name='inputs-x'
        )
        self.inputs_y = tf.placeholder(
            tf.int32, [None, input_rows, input_cols],
            name='inputs-y'
        )
        # # encoder
        ds1 = slim.conv2d(self.inputs_x, 64, 7, strides=2, padding='same')
        # slim.conv2d has default activation relu
        ds2 = slim.max_pool2d(ds1, 2)
        unet1 = self.multi_res_block(ds2, 3, 64, )





        # [224, 224, 3]
        # encoder
        b1_output = self.res_block(
            self.inputs_x, 'res-block-1', 3, 64, down_sampling=False,
            is_training=is_training)
        # [224, 224, 64]
        b2_output = self.res_block(
            b1_output, 'res-block-2', 64, 128, down_sampling=True,
            is_training=is_training)
        # [112, 112, 128]
        b3_output = self.res_block(
            b2_output, 'res-block-3', 128, 256, down_sampling=True,
            is_training=is_training)
        # [56, 56, 256]
        # bridge
        b4_output = self.res_block(
            b3_output, 'res-block-4', 256, 512, down_sampling=True,
            is_training=is_training)
        # [28, 28, 512]
        b4_up = tf.keras.layers.UpSampling2D(size=(2, 2))(b4_output)
        # [56, 56, 512] ... weird

        # decoder
        b5_input = tf.concat([b4_up, b3_output], axis=-1)
        # [56, 56, 768] ... weird
        b5_output = self.res_block(
            b5_input, 'res-block-5', 768, 256, down_sampling=False,
            is_training=is_training)
        b5_up = tf.keras.layers.UpSampling2D(size=(2, 2))(b5_output)
        # [112, 112, 256]

        b6_input = tf.concat([b5_up, b2_output], axis=-1)
        # [112, 112, 384]
        b6_output = self.res_block(
            b6_input, 'res-block-6', 384, 128, down_sampling=False,
            is_training=is_training)
        b6_up = tf.keras.layers.UpSampling2D(size=(2, 2))(b6_output)
        # [224, 224, 128]

        b7_input = tf.concat([b6_up, b1_output], axis=-1)
        # [224, 224, 192]
        b7_output = self.res_block(
            b7_input, 'res-block-7', 192, 64, down_sampling=False,
            is_training=is_training)
        # [224, 224, 64]

        w_fin = tf.get_variable('weight-fin', [1, 1, 64, 1])
        conv_fin = tf.nn.conv2d(
            b7_output, w_fin, strides=[1, 1, 1, 1], padding='SAME')
        # [224, 224, 1]

        logits = tf.reshape(conv_fin, [-1, input_rows * input_cols])
        self.score = tf.nn.sigmoid(logits)
        self.pred = tf.cast(
            tf.greater_equal(tf.nn.sigmoid(logits), 0.5), tf.int32
        )
        labels = tf.reshape(self.inputs_y, [-1, input_rows * input_cols])
        diff = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32),
            logits=logits
        )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(diff)
        with tf.variable_scope('acc'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(labels, self.pred), tf.float32)
            )
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)


if __name__ == '__main__':
    # unit test
    settings = Settings()
    with tf.Session() as sess:
        model = ResUNet(settings, is_training=True)
        tot = 0
        for var in tf.trainable_variables():
            # print(var)
            shape = var.get_shape().as_list()
            tmp = 1
            for dim in shape:
                tmp *= dim
            tot += tmp
        print('Total count of trainables:', tot)
        print(model.loss)
        print(model.accuracy)
