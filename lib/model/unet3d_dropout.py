import tensorflow as tf
from .normalization import group_norm


class SegEngine3D:
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.is_train = params['mode']
        self.layer_sampling = 2
        self.num_classes = params['num_classes']
        self.activation = tf.nn.relu

        self.drop_rate = tf.cond(self.is_train, lambda: tf.constant(0.6), lambda: tf.constant(1.0))

    def group_norm(self, inputs, is_training=True, name='group_norm'):
        return group_norm(inputs, scope=name)

    def batch_norm(self, inputs, is_training=True, name="batch_norm"):
        train_bn = tf.contrib.layers.batch_norm(inputs=inputs, scale=True, decay=0.9, updates_collections=None,
                                        fused=True, is_training=True, zero_debias_moving_mean=True, scope=name)
        test_bn = tf.contrib.layers.batch_norm(inputs=inputs, scale=True, decay=0.9, updates_collections=None,
                                       fused=True, is_training=False, reuse=True, zero_debias_moving_mean=True, scope=name)

        return tf.cond(is_training, lambda: train_bn, lambda: test_bn)

    def conv3d(self, inputs, kernel_size=(3, 3, 3), filter_size=8, name="conv_block"):
        with tf.variable_scope(name):
            conv = tf.layers.conv3d(inputs=inputs, filters=filter_size,
                                    kernel_size=kernel_size, padding="same",
                                    activation=None, name="conv1")
            # group_norm = self.group_norm(conv, self.is_train, name='group_norm')
            batch_norm = self.batch_norm(conv, self.is_train)

            act = self.activation(batch_norm)
        return act

    def deconv(self, inputs, kernel_size=(2, 2, 2), filter_size=None, name="deconv_block"):
        with tf.variable_scope(name):
            deconv = tf.layers.conv3d_transpose(inputs=inputs, filters=filter_size,
                                                kernel_size=kernel_size, padding="same",
                                                activation=None, name="deconv",
                                                strides=[2, 2, 1],
                                                use_bias=False,
                                                bias_initializer=tf.zeros_initializer())
            # group_norm = self.group_norm(deconv, self.is_train, name='group_norm')
            # batch_norm = tf.layers.batch_normalization(inputs=deconv,
            #                                            training=self.is_train,
            #                                            momentum=0.9,
            #                                            renorm_momentum=0.9)
            #
            # act = self.activation(batch_norm)
        return deconv

    def pooling(self, conv, pool_type='max_pool', name='pool'):
        if pool_type == 'max_pool':
            pool = tf.layers.max_pooling3d(inputs=conv, pool_size=[2, 2, 2],
                                           padding="same", name="max_"+name, strides=[2, 2, 1])
        elif pool_type == 'str_pool':
            filter_size = conv.get_shape().as_list()[-1]
            pool = tf.layers.conv3d(inputs=conv, filters=filter_size, kernel_size=(2, 2, 2),
                                    strides=(2, 2, 1), padding="same", activation=None, name="str_"+name)
        return pool

    def last_block(self, inputs, name="last_conv"):
        bottleneck = tf.layers.conv3d(inputs=inputs, filters=self.num_classes,
                                      kernel_size=[1, 1, 1], padding="same",
                                      activation=self.activation, name=name)
        return bottleneck

    def residual_block(self, input_layer, filter_size=32, name="residual_block"):
        conv = self.conv3d(input_layer, filter_size=filter_size, name=name + '_1')
        conv = tf.nn.dropout(conv, keep_prob=self.drop_rate)
        conv = self.conv3d(conv, filter_size=filter_size, name=name + '_2')
        return conv

    def inference(self, input_layer):
        with tf.variable_scope("down"):
            conv1 = self.residual_block(input_layer, filter_size=32, name="res_block_1")
            pool1 = self.pooling(conv1, pool_type='str_pool', name='pool_1')
            conv2 = self.residual_block(pool1, filter_size=64, name="res_block_2")
            pool2 = self.pooling(conv2, pool_type='str_pool', name='pool_2')

        with tf.variable_scope("middle"):
            mid = self.residual_block(pool2, filter_size=128, name="mid_block")

        with tf.variable_scope("upscaling"):
            up1 = self.deconv(mid, filter_size=64, name="deconv_block_1")
            up1 = tf.concat([conv2, up1], axis=4)
            conv4 = self.residual_block(up1, filter_size=64, name="res_block_4")

            up2 = self.deconv(conv4, filter_size=32, name="deconv_block_2")
            up2 = tf.concat([conv1, up2], axis=4)
            conv5 = self.residual_block(up2, filter_size=32, name="res_block_5")

        with tf.variable_scope("last_block"):
            logit = self.last_block(conv5, name="last_conv")
        return logit
