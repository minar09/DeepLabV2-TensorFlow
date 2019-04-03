from __future__ import print_function
from utils import *
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import EvalMetrics
import denseCRF
from utils import BatchDatsetReader
from utils import read_10k_data
from utils import read_LIP_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hide the warning messages about CPU/GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DATA_SET = "LIP"
DATA_SET = "10k"
# DATA_SET = "CFPD"

N_CLASSES = 20
IMAGE_DIR = 'D:/Datasets/LIP/validation/images/'
LABEL_DIR = 'D:/Datasets/LIP/validation/labels/'
NUM_STEPS = 10000  # Number of images in the validation set.
RESTORE_FROM = './checkpoint/deeplabv2_LIP'
OUTPUT_DIR = './output/deeplabv2_LIP/'
train_records, valid_records = read_LIP_data.read_dataset(
    IMAGE_DIR)
INPUT_SIZE = (384, 384)
IMAGE_SIZE = 384

if DATA_SET == "10k":
    N_CLASSES = 18
    IMAGE_DIR = 'D:/Datasets/Dressup10k/images/validation/'
    NUM_STEPS = 1000  # Number of images in the validation set.
    RESTORE_FROM = './checkpoint/deeplabv2_10k'
    OUTPUT_DIR = './output/deeplabv2_10k/'
    train_records, valid_records = read_10k_data.read_dataset(IMAGE_DIR)
    INPUT_SIZE = (224, 224)
    IMAGE_SIZE = 224

elif DATA_SET == "CFPD":
    N_CLASSES = 23
    IMAGE_DIR = 'D:/Datasets/CFPD/valimages/'
    NUM_STEPS = 536  # Number of images in the validation set.
    RESTORE_FROM = './checkpoint/deeplabv2_CFPD'
    OUTPUT_DIR = './output/deeplabv2_CFPD/'
    INPUT_SIZE = (224, 224)
    IMAGE_SIZE = 224

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = DataSetReader(IMAGE_DIR, LABEL_DIR, DATA_SET,
                               INPUT_SIZE, False, False, False, coord, DATA_SET)
        # reader = DataSetReader(IMAGE_DIR, coord, DATA_SET)
        image = reader.image
        label = reader.label
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list
        label_list = reader.label_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(
        image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(
        image_batch_origin, [int(h * 1.25), int(w * 1.25)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = DeepLabV2Model({'data': image_batch},
                                 is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = DeepLabV2Model({'data': image_batch075},
                                 is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = DeepLabV2Model({'data': image_batch125},
                                 is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(
                                                parsing_out1_075, tf.shape(image_batch_origin)[1:3, ]),
                                            tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)

    raw_output = tf.reduce_mean(
        tf.stack([parsing_out1]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=N_CLASSES, axis=2)
    tail_list_rev = [None] * N_CLASSES

    if DATA_SET == "LIP":
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx]

        tail_list_rev[14] = tail_list[15]
        tail_list_rev[15] = tail_list[14]
        tail_list_rev[16] = tail_list[17]
        tail_list_rev[17] = tail_list[16]
        tail_list_rev[18] = tail_list[19]
        tail_list_rev[19] = tail_list[18]

    elif DATA_SET == "10k":
        for xx in range(9):
            tail_list_rev[xx] = tail_list[xx]

        tail_list_rev[9] = tail_list[10]
        tail_list_rev[10] = tail_list[9]
        tail_list_rev[11] = tail_list[11]
        tail_list_rev[12] = tail_list[13]
        tail_list_rev[13] = tail_list[12]
        tail_list_rev[14] = tail_list[15]
        tail_list_rev[15] = tail_list[14]
        tail_list_rev[16] = tail_list[16]
        tail_list_rev[17] = tail_list[17]

    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(
        tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    logits = raw_output_all
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    # Create 4-d tensor.
    prediction_all = tf.expand_dims(raw_output_all, dim=3)

    # Which variables to load.
    restore_var = tf.global_variables()
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

    cross_mats = list()
    crf_cross_mats = list()

    probability = tf.nn.softmax(logits=logits, axis=3)

    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    validation_dataset_reader = BatchDatsetReader.BatchDatset(
        valid_records, image_options)
    validation_dataset_reader.reset_batch_offset(0)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        try:
            parsing_, probpred = sess.run([prediction_all, probability])
            if step % 100 == 0:
                print('step {:d}'.format(step))
                print(image_list[step])
            img_split = image_list[step].split('/')
            # extra split for modified file readers
            img_split = img_split[-1].split('\\')[1]
            img_id = img_split[:-4]

            msk = decode_labels(parsing_, num_classes=N_CLASSES)
            parsing_im = Image.fromarray(msk[0])
            parsing_im.save('{}/pred_{}_vis.png'.format(OUTPUT_DIR, img_id))
            cv2.imwrite('{}/pred_{}.png'.format(OUTPUT_DIR, img_id),
                        parsing_[0, :, :, 0])

            try:
                inp, gt = validation_dataset_reader.next_batch(1)

                msk = decode_labels(gt, num_classes=N_CLASSES)
                parsing_im = Image.fromarray(msk[0])
                parsing_im.save('{}/gt_{}_vis.png'.format(OUTPUT_DIR, img_id))
                cv2.imwrite('{}/gt_{}.png'.format(OUTPUT_DIR, img_id),
                            gt[0, :, :, 0])

                inp = inp[0]
                gt = gt[0]
                gt = np.squeeze(gt, axis=2)

                # Confusion matrix for this image prediction
                crossMat = EvalMetrics.calculate_confusion_matrix(
                    gt.astype(
                        np.uint8), parsing_[0, :, :, 0].astype(
                        np.uint8), N_CLASSES)
                cross_mats.append(crossMat)

                np.savetxt(OUTPUT_DIR +
                           "Crossmatrix" +
                           str(img_id) +
                           ".csv", crossMat, fmt='%4i', delimiter=',')

                # Save input, gt, pred, crf_pred, sum figures for this image

                """ Generate CRF """
                # 1. run CRF
                crfwithprobsoutput = denseCRF.crf_with_probs(
                    inp.astype(np.uint8), probpred[0], N_CLASSES)

                # 2. show result display
                crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

                crfwithprobspred_expanded = np.expand_dims(
                    crfwithprobspred, axis=0)
                crfwithprobspred_expanded = np.expand_dims(
                    crfwithprobspred_expanded, axis=3)

                msk = decode_labels(
                    crfwithprobspred_expanded, num_classes=N_CLASSES)
                parsing_im = Image.fromarray(msk[0])
                parsing_im.save(
                    '{}/crf_{}_vis.png'.format(OUTPUT_DIR, img_id))
                cv2.imwrite('{}/crf_{}.png'.format(OUTPUT_DIR, img_id),
                            crfwithprobspred)

                # Confusion matrix for this image prediction with crf
                prob_crf_crossMat = EvalMetrics.calculate_confusion_matrix(
                    gt.astype(
                        np.uint8), crfwithprobsoutput.astype(
                        np.uint8), N_CLASSES)
                crf_cross_mats.append(prob_crf_crossMat)

                np.savetxt(OUTPUT_DIR +
                           "crf_Crossmatrix" +
                           str(img_id) +
                           ".csv", prob_crf_crossMat, fmt='%4i', delimiter=',')

            except Exception as e:
                print(e)

        except Exception as err:
            print(err)

    try:
        total_cm = np.sum(cross_mats, axis=0)
        np.savetxt(
            OUTPUT_DIR +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        print(">>> Prediction results:")
        EvalMetrics.show_result(total_cm, N_CLASSES)

        # Prediction with CRF
        crf_total_cm = np.sum(crf_cross_mats, axis=0)
        np.savetxt(
            OUTPUT_DIR +
            "CRF_Crossmatrix.csv",
            crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print(">>> Prediction results (CRF (prob)):")
        EvalMetrics.show_result(crf_total_cm, N_CLASSES)

    except Exception as err:
        print(err)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
