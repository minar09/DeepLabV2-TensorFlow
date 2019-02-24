"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import time
import os
import random
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label, load, save
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hide the warning messages about CPU/GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

BATCH_SIZE = 4
IMAGE_DIR = 'D:/Datasets/Dressup10k/images/training/'
LABEL_DIR = 'D:/Datasets/Dressup10k/annotations/training/'
GRAD_UPDATE_EVERY = 10
IGNORE_LABEL = 255
INPUT_SIZE = [321, 321]
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 18
NUM_IMAGES = 9003
NUM_EPOCHS = 30
SAVE_PRED_EVERY = NUM_IMAGES // BATCH_SIZE
NUM_STEPS = NUM_EPOCHS * SAVE_PRED_EVERY
POWER = 0.9
RESTORE_FROM = './checkpoints/deeplab_resnet_10k/'
SAVE_NUM_IMAGES = 1
SNAPSHOT_DIR = './logs/deeplab_resnet_10k/'
WEIGHT_DECAY = 0.0005
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
IS_TRAINING = True
NOT_RESTORE_LAST = False


def main():
    """Create the model and start the training."""

    h, w = INPUT_SIZE

    random_seed = random.randint(1000, 9999)
    tf.set_random_seed(random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            IMAGE_DIR,
            LABEL_DIR,
            INPUT_SIZE,
            RANDOM_SCALE,
            RANDOM_MIRROR,
            IGNORE_LABEL,
            IMG_MEAN,
            coord,
            SHUFFLE)
        image_batch, label_batch = reader.dequeue(BATCH_SIZE)
        image_batch075 = tf.image.resize_images(
            image_batch, [int(h * 0.75), int(w * 0.75)])
        image_batch05 = tf.image.resize_images(
            image_batch, [int(h * 0.5), int(w * 0.5)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel(
            {'data': image_batch}, is_training=IS_TRAINING, num_classes=NUM_CLASSES)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel(
            {'data': image_batch075}, is_training=IS_TRAINING, num_classes=NUM_CLASSES)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel(
            {'data': image_batch05}, is_training=IS_TRAINING, num_classes=NUM_CLASSES)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = net075.layers['fc1_voc12']
    raw_output05 = net05.layers['fc1_voc12']
    raw_output = tf.reduce_max(tf.stack([raw_output100,
                                         tf.image.resize_images(
                                             raw_output075, tf.shape(raw_output100)[1:3, ]),
                                         tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3, ])]), axis=0)
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables(
    ) if 'fc' not in v.name or not NOT_RESTORE_LAST]
    all_trainable = [v for v in tf.trainable_variables(
    ) if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [
        v for v in all_trainable if 'fc' not in v.name]  # lr * 1.0
    fc_w_trainable = [
        v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [
        v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, NUM_CLASSES])
    raw_prediction100 = tf.reshape(raw_output100, [-1, NUM_CLASSES])
    raw_prediction075 = tf.reshape(raw_output075, [-1, NUM_CLASSES])
    raw_prediction05 = tf.reshape(raw_output05, [-1, NUM_CLASSES])

    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape(
    )[1:3]), num_classes=NUM_CLASSES, one_hot=False)  # [batch_size, h, w]
    label_proc075 = prepare_label(label_batch, tf.stack(raw_output075.get_shape()[
                                  1:3]), num_classes=NUM_CLASSES, one_hot=False)
    label_proc05 = prepare_label(label_batch, tf.stack(raw_output05.get_shape()[
                                 1:3]), num_classes=NUM_CLASSES, one_hot=False)

    raw_gt = tf.reshape(label_proc, [-1, ])
    raw_gt075 = tf.reshape(label_proc075, [-1, ])
    raw_gt05 = tf.reshape(label_proc05, [-1, ])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1)), 1)
    indices075 = tf.squeeze(
        tf.where(tf.less_equal(raw_gt075, NUM_CLASSES - 1)), 1)
    indices05 = tf.squeeze(
        tf.where(tf.less_equal(raw_gt05, NUM_CLASSES - 1)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
    gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)

    prediction = tf.gather(raw_prediction, indices)
    prediction100 = tf.gather(raw_prediction100, indices)
    prediction075 = tf.gather(raw_prediction075, indices075)
    prediction05 = tf.gather(raw_prediction05, indices05)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=gt)
    loss100 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction100, labels=gt)
    loss075 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction075, labels=gt075)
    loss05 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction05, labels=gt05)
    l2_losses = [WEIGHT_DECAY *
                 tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(
        loss075) + tf.reduce_mean(loss05) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(
        raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Image summary.
    images_summary = tf.py_func(
        inv_preprocess, [image_batch, SAVE_NUM_IMAGES, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(
        decode_labels, [label_batch, SAVE_NUM_IMAGES, NUM_CLASSES], tf.uint8)
    preds_summary = tf.py_func(
        decode_labels, [pred, SAVE_NUM_IMAGES, NUM_CLASSES], tf.uint8)

    total_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[
                                               images_summary, labels_summary, preds_summary]),
                                     max_outputs=SAVE_NUM_IMAGES)  # Concatenate row-wise.
    summary_writer = tf.summary.FileWriter(SNAPSHOT_DIR,
                                           graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(
        base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)

    # Define a variable to accumulate gradients.
    accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                               trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]

    # Define an operation to clear the accumulated gradients for next batch.
    zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

    # Compute gradients.
    grads = tf.gradients(reduced_loss, conv_trainable +
                         fc_w_trainable + fc_b_trainable)

    # Accumulate and normalise the gradients.
    accum_grads_op = [accum_grads[i].assign_add(
        grad / GRAD_UPDATE_EVERY) for i, grad in enumerate(grads)]

    grads_conv = accum_grads[:len(conv_trainable)]
    grads_fc_w = accum_grads[len(conv_trainable): (
        len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]

    # Apply the gradients.
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    # Load variables if the checkpoint is provided.
    if RESTORE_FROM is not None:
        loader = tf.train.Saver(var_list=restore_var)
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict = {step_ph: step}
        loss_value = 0

        # Clear the accumulated gradients.
        sess.run(zero_op, feed_dict=feed_dict)

        # Accumulate gradients.
        for i in range(GRAD_UPDATE_EVERY):
            _, l_val = sess.run(
                [accum_grads_op, reduced_loss], feed_dict=feed_dict)
            loss_value += l_val

        # Normalise the loss.
        loss_value /= GRAD_UPDATE_EVERY

        # Apply gradients.
        if step % SAVE_PRED_EVERY == 0 and step > 0:
            images, labels, summary, _ = sess.run(
                [image_batch, label_batch, total_summary, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, SNAPSHOT_DIR, step)
        else:
            sess.run(train_op, feed_dict=feed_dict)

        duration = time.time() - start_time
        print(
            'step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
