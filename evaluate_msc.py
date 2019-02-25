"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from deeplab_resnet import DeepLabResNetModel, ImageReader, load, decode_labels

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hide the warning messages about CPU/GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

IMAGE_DIR = 'D:/Datasets/Dressup10k/images/validation/'
LABEL_DIR = 'D:/Datasets/Dressup10k/annotations/validation/'
IGNORE_LABEL = 255
NUM_CLASSES = 18
NUM_STEPS = 1000  # Number of images in the validation set.
RESTORE_FROM = './logs/deeplab_resnet_10k/'
OUTPUT_DIR = './output/deeplab_resnet_10k/'


def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            IMAGE_DIR,
            LABEL_DIR,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            IGNORE_LABEL,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
        image_list = reader.image_list

    # Add one batch dimension.
    image_batch, label_batch = tf.expand_dims(
        image, dim=0), tf.expand_dims(label, dim=0)
    h_orig, w_orig = tf.to_float(
        tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch075 = tf.image.resize_images(image_batch, tf.stack(
        [tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch05 = tf.image.resize_images(image_batch, tf.stack(
        [tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel(
            {'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel(
            {'data': image_batch075}, is_training=False, num_classes=NUM_CLASSES)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel(
            {'data': image_batch05}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = tf.image.resize_images(
        net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3, ])
    raw_output05 = tf.image.resize_images(
        net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3, ])

    raw_output = tf.reduce_max(
        tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    raw_output = tf.image.resize_bilinear(
        raw_output, tf.shape(image_batch)[1:3, ])
    raw_output = tf.argmax(raw_output, dimension=3)
    prediction_all = tf.expand_dims(raw_output, dim=3)  # Create 4-d tensor.

    # mIoU
    prediction = tf.reshape(prediction_all, [-1, ])
    gt = tf.reshape(label_batch, [-1, ])
    # Ignoring all labels greater than or equal to n_classes.
    weights = tf.cast(tf.less_equal(gt, NUM_CLASSES - 1), tf.int32)
    mean_iou, update_op = tf.contrib.metrics.streaming_mean_iou(
        prediction, gt, num_classes=NUM_CLASSES, weights=weights)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Iterate over training steps.
    for step in range(NUM_STEPS):

        predictions, _ = sess.run([prediction_all, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))

        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        msk = decode_labels(predictions, num_classes=NUM_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        parsing_im.save('{}/{}_vis.png'.format(OUTPUT_DIR, img_id))
        cv2.imwrite('{}/{}.png'.format(OUTPUT_DIR, img_id),
                    predictions[0, :, :, 0])

    print('Mean IoU: {:.3f}'.format(mean_iou.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
