import os
from glob import glob
import numpy as np
import shutil
import tensorflow as tf

from model.unet3d import SegEngine3D
from builder import input_builder
from util.general_util import progress_bar
from core.loss_segmentation import LossFunction


def save_image_summary_3d(image, name='image'):
    max_outputs = 1

    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image)) * 255
    image = tf.cast(image, tf.uint8)

    image_shape = image.get_shape().as_list()
    image_shape[3] = 1
    image_shape[0] = max_outputs

    sliced_image = tf.slice(image,
                            [0, 0, 0, 0],
                                tf.stack(image_shape),
                            name='sliced_%s' % name)

    # sliced_image = tf.expand_dims(sliced_image, axis=3)
    tf.summary.image(name=name, tensor=tf.cast(sliced_image, tf.uint8), max_outputs=max_outputs)


num_class = 3
num_epochs = 200
batch_size = 4
split_type = 'kfolds'

learning_rate = 0.0001

save_log = 'c://workspace/3. Contest/ISLES2018/log_0801/'
if os.path.exists(save_log):
    shutil.rmtree(save_log)

with tf.Graph().as_default():
    input_image = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 4, 1], name='input_image')
    groundtruth = tf.placeholder(dtype=tf.uint8, shape=[None, 256, 256, 4], name='groundtruth')
    is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')

    params = {'batch_size': batch_size,
              'num_classes': num_class,
              'mode': is_training}

    logit = SegEngine3D(params).inference(input_image)

    with tf.variable_scope("image"):
        save_image_summary_3d(tf.squeeze(input_image, 4), name='image')
        save_image_summary_3d(tf.argmax(logit, 4), name='logit')
        save_image_summary_3d(groundtruth, name='groundtruth')

    with tf.variable_scope('Loss'):
        loss_fn = LossFunction(n_class=num_class, loss_type='Dice')
        loss = loss_fn(prediction=logit, ground_truth=groundtruth)

    global_step = tf.train.get_global_step()
    with tf.variable_scope('Optimization'):
        # lr = tf.train.exponential_decay(learning_rate=learning_rate,
        #                                 global_step=global_step,
        #                                 decay_steps=30000,
        #                                 decay_rate=0.5,
        #                                 staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    eval_ops = {'loss': loss}
    [tf.summary.scalar(eval_name, eval_ops[eval_name]) for eval_name in eval_ops.keys()]

    data_input = input_builder.InputData(batch_size=batch_size, num_class=num_class)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False,
                                 gpu_options=tf.GPUOptions(
                                     force_gpu_compatible=True,
                                     allow_growth=True))

    merge = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(save_log, sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(save_log, 'eval'), sess.graph)

        glob_step = 0
        cost_total = 0

        print('Start training')
        for epoch_n in range(num_epochs):
            data_input.split_trainset(epoch_n, split_type)

            for batch_n in range(data_input.batches_per_epoch):
                image, label = data_input.generate_dataset(batch_n)

                feed_dict = {input_image: image, groundtruth: label}
                _, cost_value = sess.run([train_op, loss], feed_dict)

                cost_total += cost_value
                cost_avg = cost_total / (glob_step + 1)

                summary = sess.run(merge, feed_dict)
                writer.add_summary(summary, glob_step)
                glob_step += 1

                # Eval per Step
                eval_batch = glob_step % data_input.eval_per_epoch
                eval_image, eval_label = data_input.generate_dataset(eval_batch, is_training=False)

                eval_feed_dict = {input_image: eval_image, groundtruth: eval_label, is_training: False}
                cost_value = sess.run(loss, eval_feed_dict)

                summary = sess.run(merge, eval_feed_dict)
                eval_writer.add_summary(summary, glob_step)

                # Progress Bar
                progress_bar(data_input.batches_per_epoch, batch_n+1,
                             '[%d/%d] Epoch.%d - cost value : %f' %
                             (batch_n+1, data_input.batches_per_epoch, epoch_n, cost_avg))

