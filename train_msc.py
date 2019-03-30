from __future__ import print_function
import os
import time
import random
from utils import *
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set gpus
gpus = [0, 1, 2, 3]  # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
NUM_GPU = len(gpus)  # number of GPUs to use

# Hide the warning messages about CPU/GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DATA_SET = "LIP"
DATA_SET = "10k"
# DATA_SET = "CFPD"

# parameters setting
N_CLASSES = 20
BATCH_SIZE = 8
BATCH_ITERATION = BATCH_SIZE // NUM_GPU
NUM_IMAGES = 30462
IMAGE_DIR = 'D:/Datasets/LIP/training/images/'
LABEL_DIR = 'D:/Datasets/LIP/training/labels/'
SNAPSHOT_DIR = './checkpoint/deeplabv2_LIP'
LOG_DIR = './logs/deeplabv2_LIP'

if DATA_SET == "10k":
    N_CLASSES = 18
    NUM_IMAGES = 9003
    IMAGE_DIR = 'D:/Datasets/Dressup10k/images/training/'
    LABEL_DIR = 'D:/Datasets/Dressup10k/annotations/training/'
    SNAPSHOT_DIR = './checkpoint/deeplabv2_10k'
    LOG_DIR = './logs/deeplabv2_10k'

elif DATA_SET == "CFPD":
    N_CLASSES = 23
    NUM_IMAGES = 1674
    IMAGE_DIR = 'D:/Datasets/CFPD/trainimages/'
    LABEL_DIR = 'D:/Datasets/CFPD/trainimages/'
    SNAPSHOT_DIR = './checkpoint/deeplabv2_CFPD'
    LOG_DIR = './logs/deeplabv2_CFPD'

INPUT_SIZE = (384, 384)
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
POWER = 0.9
SAVE_PREDICTION_EVERY = NUM_IMAGES // BATCH_SIZE
NUM_EPOCHS = 30
NUM_STEPS = SAVE_PREDICTION_EVERY * NUM_EPOCHS
SHOW_STEP = 10


def main():
    random_seed = random.randint(1000, 9999)
    tf.set_random_seed(random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = DataSetReader(IMAGE_DIR, LABEL_DIR, DATA_SET,
                               INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
        image_batch, label_batch = reader.dequeue(BATCH_SIZE)
        image_batch075 = tf.image.resize_images(
            image_batch, [int(h * 0.75), int(w * 0.75)])
        image_batch050 = tf.image.resize_images(
            image_batch, [int(h * 0.5), int(w * 0.5)])

    tower_grads = []

    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(
        base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))
    optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)

    reduced_loss = None

    for i in range(NUM_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % i) as scope:

                if i == 0:
                    reuse1 = False
                    reuse2 = True
                else:
                    reuse1 = True
                    reuse2 = True

                next_image = image_batch[i *
                                         BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                next_image075 = image_batch075[i *
                                               BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                next_image050 = image_batch050[i *
                                               BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                next_label = label_batch[i *
                                         BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]

                # Create network.
                with tf.variable_scope('', reuse=reuse1):
                    net_100 = DeepLabV2Model(
                        {'data': next_image}, is_training=False, n_classes=N_CLASSES)
                with tf.variable_scope('', reuse=reuse2):
                    net_075 = DeepLabV2Model(
                        {'data': next_image075}, is_training=False, n_classes=N_CLASSES)
                with tf.variable_scope('', reuse=reuse2):
                    net_050 = DeepLabV2Model(
                        {'data': next_image050}, is_training=False, n_classes=N_CLASSES)

                # parsing net
                parsing_out1_100 = net_100.layers['fc1_human']
                parsing_out1_075 = net_075.layers['fc1_human']
                parsing_out1_050 = net_050.layers['fc1_human']

                # combine resize
                parsing_out1 = tf.reduce_mean(tf.stack([parsing_out1_100,
                                                        tf.image.resize_images(parsing_out1_075,
                                                                               tf.shape(parsing_out1_100)[1:3, ]),
                                                        tf.image.resize_images(parsing_out1_050,
                                                                               tf.shape(parsing_out1_100)[1:3, ])]),
                                              axis=0)

                # Predictions: ignoring all predictions with labels greater or equal than n_classes
                raw_prediction_p1 = tf.reshape(parsing_out1, [-1, N_CLASSES])
                raw_prediction_p1_100 = tf.reshape(
                    parsing_out1_100, [-1, N_CLASSES])
                raw_prediction_p1_075 = tf.reshape(
                    parsing_out1_075, [-1, N_CLASSES])
                raw_prediction_p1_050 = tf.reshape(
                    parsing_out1_050, [-1, N_CLASSES])

                label_proc = prepare_label(next_label, tf.stack(parsing_out1.get_shape()[1:3]),
                                           one_hot=False, num_classes=N_CLASSES)  # [batch_size, h, w]
                label_proc075 = prepare_label(next_label, tf.stack(
                    parsing_out1_075.get_shape()[1:3]), one_hot=False, num_classes=N_CLASSES)
                label_proc050 = prepare_label(next_label, tf.stack(
                    parsing_out1_050.get_shape()[1:3]), one_hot=False, num_classes=N_CLASSES)

                raw_gt = tf.reshape(label_proc, [-1, ])
                raw_gt075 = tf.reshape(label_proc075, [-1, ])
                raw_gt050 = tf.reshape(label_proc050, [-1, ])

                indices = tf.squeeze(
                    tf.where(tf.less_equal(raw_gt, N_CLASSES - 1)), 1)
                indices075 = tf.squeeze(
                    tf.where(tf.less_equal(raw_gt075, N_CLASSES - 1)), 1)
                indices050 = tf.squeeze(
                    tf.where(tf.less_equal(raw_gt050, N_CLASSES - 1)), 1)

                gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
                gt050 = tf.cast(tf.gather(raw_gt050, indices050), tf.int32)

                prediction_p1 = tf.gather(raw_prediction_p1, indices)
                prediction_p1_100 = tf.gather(raw_prediction_p1_100, indices)
                prediction_p1_075 = tf.gather(
                    raw_prediction_p1_075, indices075)
                prediction_p1_050 = tf.gather(
                    raw_prediction_p1_050, indices050)

                # Pixel-wise softmax loss.
                loss_p1 = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1, labels=gt))
                loss_p1_100 = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_100, labels=gt))
                loss_p1_075 = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_075, labels=gt075))
                loss_p1_050 = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_050, labels=gt050))

                reduced_loss = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050

                trainable_variable = tf.trainable_variables()
                grads = optimizer.compute_gradients(
                    reduced_loss, var_list=trainable_variable)

                tower_grads.append(grads)

                tf.add_to_collection('loss_p1', loss_p1)
                tf.add_to_collection('loss_p1_100', loss_p1_100)
                tf.add_to_collection('loss_p1_075', loss_p1_075)
                tf.add_to_collection('loss_p1_050', loss_p1_050)
                tf.add_to_collection('reduced_loss', reduced_loss)

    # Average the gradients
    grads_ave = average_gradients(tower_grads)
    # apply the gradients with our optimizers
    train_op = optimizer.apply_gradients(grads_ave)

    loss_p1_ave = tf.reduce_mean(tf.get_collection('loss_p1'))
    loss_p1_100_ave = tf.reduce_mean(tf.get_collection('loss_p1_100'))
    loss_p1_075_ave = tf.reduce_mean(tf.get_collection('loss_p1_075'))
    loss_p1_050_ave = tf.reduce_mean(tf.get_collection('loss_p1_050'))
    loss_ave = tf.reduce_mean(tf.get_collection('reduced_loss'))

    loss_summary_p1 = tf.summary.scalar("loss_p1_ave", loss_p1_ave)
    loss_summary_p1_100 = tf.summary.scalar("loss_p2_ave", loss_p1_100_ave)
    loss_summary_p1_075 = tf.summary.scalar("loss_p3_ave", loss_p1_075_ave)
    loss_summary_p1_050 = tf.summary.scalar("loss_s1_ave", loss_p1_050_ave)
    loss_summary_ave = tf.summary.scalar("loss_ave", loss_ave)
    loss_summary = tf.summary.merge(
        [loss_summary_ave, loss_summary_p1_100, loss_summary_p1_075, loss_summary_p1_050, loss_summary_p1])
    summary_writer = tf.summary.FileWriter(
        LOG_DIR, graph=tf.get_default_graph())

    # Set up tf session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    all_saver_var = tf.global_variables()
    # [v for v in all_saver_var if 'pose' not in v.name and 'parsing' not in v.name]
    restore_var = all_saver_var
    saver = tf.train.Saver(var_list=all_saver_var, max_to_keep=1)
    loader = tf.train.Saver(var_list=restore_var)

    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict = {step_ph: step}

        # Apply gradients.
        summary, loss_value, _ = sess.run(
            [loss_summary, reduced_loss, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)

        if step % SAVE_PREDICTION_EVERY == 0 and step > 0:
            save(saver, sess, SNAPSHOT_DIR, step)

        if step % SHOW_STEP == 0:
            duration = time.time() - start_time
            print(
                'step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    main()
