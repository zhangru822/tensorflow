# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# 从CIFAR10数据文件中读取样例
# filename_queue一个队列的文件名
def read_cifar10(filename_queue):


    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    # 分类结果的长度，CIFAR-100长度为2
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    # 3位表示rgb颜色（0-255,0-255,0-255）
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    # 单个记录的总长度=分类结果长度+图片长度
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    # 读取
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 第一位代表lable-图片的正确分类结果，从uint8转换为int32类型
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 分类结果之后的数据代表图片，我们重新调整大小
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # 格式转换，从[颜色,高度,宽度]--》[高度,宽度,颜色]
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

# 构建一个排列后的一组图片和分类
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    # 线程数
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])



def _get_images_labels(batch_size, split, distords=False):
  """Returns Dataset for given split."""
  dataset = tfds.load(name='cifar10', split=split)
  scope = 'data_augmentation' if distords else 'input'
  with tf.name_scope(scope):
    dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
  # Dataset is small enough to be fully loaded on memory:
  dataset = dataset.prefetch(-1)
  dataset = dataset.repeat().batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images_labels = iterator.get_next()
  images, labels = images_labels['input'], images_labels['target']
  tf.summary.image('images', images)
  return images, labels


class DataPreprocessor(object):
  """Applies transformations to dataset record."""

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):
    """Process img for training or eval."""
    img = record['image']
    img = tf.cast(img, tf.float32)
    if self._distords:  # training
      # Randomly crop a [height, width] section of the image.
      img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
      # Randomly flip the image horizontally.
      img = tf.image.random_flip_left_right(img)
      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      # NOTE: since per_image_standardization zeros the mean and makes
      # the stddev unit, this likely has no effect see tensorflow#1458.
      img = tf.image.random_brightness(img, max_delta=63)
      img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    else:  # Image processing for evaluation.
      # Crop the central [height, width] of the image.
      img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    # Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    return dict(input=img, target=record['label'])

# 为CIFAR评价构建输入
# data_dir路径
# batch_size一个组的大小
def distorted_inputs(data_dir, batch_size):
  
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    # 随机裁剪图片
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    # 随机旋转图片
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # 亮度变换
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # 对比度变换
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # Linearly scales image to have zero mean and unit norm
    # 标准化
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    # 设置张量的型
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    # 确保洗牌的随机性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

# 为CIFAR评价构建输入
# eval_data使用训练还是评价数据集
# data_dir路径
# batch_size一个组的大小
def inputs(eval_data, data_dir, batch_size):
   
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    # 文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    # 从文件中读取解析出的图片队列
    read_input = read_cifar10(filename_queue)
    # 转换为float
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    # 剪切图片的中心
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    # 标准化图片
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    # 设置张量的型
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    # 确保洗牌的随机性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
