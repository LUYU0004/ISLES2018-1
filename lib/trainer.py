import os
from glob import glob
import numpy as np

import tensorflow as tf

from model.unet3d import SegEngine3D
from builder import input_builder
from util.general_util import progress_bar
from core.loss_segmentation import LossFunction


def save_image_summary_3d(image, name='image'):
    max_outputs = 3

    image_shape = image.get_shape().as_list()
    image_shape[3] = 1
    image_shape[0] = max_outputs

    sliced_image = tf.slice(tf.cast(image, tf.uint8),
                            [0, 0, 0, 2],
                            tf.stack(image_shape),
                            name='sliced_%s' % name)

    # sliced_image = tf.expand_dims(sliced_image, axis=3)
    tf.summary.image(name=name, tensor=tf.cast(sliced_image, tf.uint8), max_outputs=max_outputs)


num_class = 2
num_epochs = 10
batch_size = 4
split_type = 'kfolds'

learning_rate = 0.0001


with tf.Graph().as_default(), tf.device('/cpu:0'):
    input_image = tf.placeholder(dtype=tf.float16, shape=[None, 256, 256, 4, 1], name='input_image')
    groundtruth = tf.placeholder(dtype=tf.float16, shape=[None, 256, 256, 4, 2], name='groundtruth')
    is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')

    params = {'batch_size': batch_size,
              'num_classes': num_class,
              'mode': is_training}

    logit = SegEngine3D(params).inference(input_image)
    save_image_summary_3d(tf.argmax(input_image, 4), name='image')
    save_image_summary_3d(tf.argmax(logit, 4), name='logit')
    # save_image_summary_3d(tf.argmax(groundtruth, 4), name='groundtruth')

    loss_fn = LossFunction(n_class=num_class, loss_type='Dice')
    cost = loss_fn(prediction=logit, ground_truth=groundtruth)
    print('cost', cost)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    eval_set = {'cost': cost}
    for eval_name in eval_set.keys():
        tf.summary.scalar(eval_name, eval_set[eval_name])

    data_input = input_builder.InputData(batch_size=batch_size, num_class=num_class)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False,
                                 gpu_options=tf.GPUOptions(
                                     force_gpu_compatible=True,
                                     allow_growth=True))

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)

        for epoch_n in range(num_epochs):
            data_input.split_trainset(epoch_n, split_type)

            for batch_n in range(data_input.batches_per_epoch):
                image, label = data_input.generate_dataset(batch_n)

                feed_dict = {input_image: image, groundtruth: label}
                _, cost_value = sess.run([train_op, cost], feed_dict)

                progress_bar(data_input.batches_per_epoch, batch_n, 'cost value : %f' % cost_value)
