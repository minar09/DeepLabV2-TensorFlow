import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

IGNORE_LABEL = 255
IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


def read_labeled_image_list(image_dir, data_set="LIP"):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    file_names = os.listdir(image_dir)
    images = []

    for line in tqdm(file_names):
        image = None

        if data_set == "CFPD":
            if ".png" in line:
                continue

        try:
            image = image_dir + line
        except ValueError:  # Adhoc for test.
            print("Error: ", line)

        images.append(image)

    return images


# optional pre-processing arguments
def read_images_from_disk(input_queue):
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    return img


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, coord, data_set="LIP"):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_set = data_set
        self.coord = coord

        self.image_list = read_labeled_image_list(
            self.data_dir, self.data_set)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images], shuffle=False)  # not shuffling if it is val
        self.image = read_images_from_disk(
            self.queue)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch = tf.train.batch([self.image], num_elements)
        return image_batch
