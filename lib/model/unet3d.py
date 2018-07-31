import tensorflow as tf
from .normalization import group_norm


class SegEngine3D:
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.is_train = params['mode']
        self.layer_sampling = 2
        self.num_classes = params['num_classes']
        self.activation = tf.nn.relu
        self.dropout_rate = 0.0

    def group_norm(self, inputs, is_training=True, name='group_norm'):
        return group_norm(inputs, scope=name)

    def batch_norm(self, inputs, is_training=True, name="batch_norm"):
        train_bn = tf.contrib.layers.batch_norm(inputs=inputs, scale=True, decay=0.9, updates_collections=None,
                                        fused=True, is_training=True, zero_debias_moving_mean=True, scope=name)
        test_bn = tf.contrib.layers.batch_norm(inputs=inputs, scale=True, decay=0.9, updates_collections=None,
                                       fused=True, is_training=False, reuse=True, zero_debias_moving_mean=True, scope=name)

        return tf.cond(is_training, lambda: train_bn, lambda: test_bn)

    def conv(self, inputs, kernel_size=(3, 3, 3), filter_size=8, name="conv_block"):
        with tf.variable_scope(name):
            conv = tf.layers.conv3d(inputs=inputs, filters=filter_size,
                                    kernel_size=kernel_size, padding="same",
                                    activation=None, name="conv1")
            group_norm = self.group_norm(conv, self.is_train, name='group_norm')
            # batch_norm = self.batch_norm(conv, self.is_train)

            # act = pRelu(batch_norm)
            act = tf.nn.relu(group_norm)
        return act

    def deconv(self, inputs, kernel_size=(2, 2, 2), filter_size=None, name="deconv_block"):
        with tf.variable_scope(name):
            deconv = tf.layers.conv3d_transpose(inputs=inputs, filters=filter_size,
                                                kernel_size=kernel_size, padding="same",
                                                activation=None, name="deconv",
                                                strides=[2, 2, 1],
                                                use_bias=False,
                                                bias_initializer=tf.zeros_initializer())
            # deconv = tf.layers.dropout(deconv, self.dropout_rate, training=self.is_train, name="dropout")
            group_norm = self.group_norm(deconv, self.is_train, name='group_norm')
            # batch_norm = tf.layers.batch_normalization(inputs=deconv,
            #                                            training=self.is_train,
            #                                            momentum=0.9,
            #                                            renorm_momentum=0.9)

            # act = pRelu(batch_norm)
            act = tf.nn.relu(group_norm)
        return act

    def residual_block(self, inputs, kernel_size=(3, 3, 3), filter_size=None, name="residual_block", pooling=True):
        with tf.variable_scope(name):
            conv_ = self.conv(inputs, kernel_size, filter_size, name="conv_1")
            # conv_ = self.conv(conv1, kernel_size, filter_size, name='conv_2')
            # conv_ = self.conv(conv_, kernel_size, filter_size, name='conv_3')
            # conv_ = self.conv(conv_, kernel_size, filter_size, name='conv_4')
            # skip_connect = tf.add(conv1, conv_, name='skip_connection')
            # conv_ = self.conv(skip_connect, kernel_size, filter_size, name="merged_conv")

            if not pooling:
                return conv_

            maxpool = tf.layers.max_pooling3d(inputs=conv_, pool_size=[2, 2, 2],
                                              padding="same", name="maxpool", strides=[2, 2, 1])

            # str_pool = tf.layers.conv3d(inputs=conv_, filters=filter_size, kernel_size=(2,2,2), strides=(2,2,2),
            #                             padding="same", activation=None, name="conv1")
            return conv_, maxpool

    def last_block(self, inputs, name="last_block"):
        with tf.variable_scope(name):
            bottleneck = tf.layers.conv3d(inputs=inputs, filters=self.num_classes,
                                          kernel_size=[1, 1, 1], padding="same",
                                          activation=self.activation, name="conv")
        return bottleneck

    def inference(self, input_layer):
        default_filters = [16, 32, 64, 128, 256]
        filters = [filter_size * self.layer_sampling for filter_size in default_filters]

        with tf.variable_scope("down"):
            down1, maxpool1 = self.residual_block(input_layer, filter_size=filters[0], name="residual_block_1")
            down2, maxpool2 = self.residual_block(maxpool1, filter_size=filters[1], name="residual_block_2")
            down3, maxpool3 = self.residual_block(maxpool2, filter_size=filters[2], name="residual_block_3")
            down4, maxpool4 = self.residual_block(maxpool3, filter_size=filters[3], name="residual_block_4")

        with tf.variable_scope("middle"):
            mid = self.residual_block(maxpool4, filter_size=filters[4], name="residual_block", pooling=False)

        with tf.variable_scope("upscaling"):
            up = tf.add(self.deconv(mid, filter_size=filters[3], name="deconv_block_1"), down4, name="merge_1")
            up_ = self.residual_block(up, filter_size=filters[3], name="residual_block_1", pooling=False)
            up = tf.add(self.deconv(up_, filter_size=filters[2], name="deconv_block_2"), down3, name="merge_2")
            up_ = self.residual_block(up, filter_size=filters[2], name="residual_block_2", pooling=False)
            up = tf.add(self.deconv(up_, filter_size=filters[1], name="deconv_block_3"), down2, name="merge_3")
            up_ = self.residual_block(up, filter_size=filters[1], name="residual_block_3", pooling=False)
            up = tf.add(self.deconv(up_, filter_size=filters[0], name="deconv_block_4"), down1, name="merge_4")
            up_ = self.residual_block(up, filter_size=filters[0], name="residual_block_4", pooling=False)

        logit = self.last_block(up_, name="last_block")
        print("up_shape :", up_.shape)
        print("logit_shape :", logit.shape)
        return logit
