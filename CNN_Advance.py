#!git clone https://github.com/tensorflow/models.git
#!cd E:\workspace\tensorflow\models\tutorials\image\cifar10

import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time

# 定义batch_size, 训练轮数max_steps, 以及下载CIFAR-10数据的默认路径
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

# 定义初始化weight的函数，定义的同时，对weight加一个L2 loss，放在集'losses'中
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 使用cifar10类下载数据集，并解压、展开到其默认位置
cifar10.maybe_download_and_extract()

# 在使用cifar10_input类中的distorted_inputs函数产生训练需要使用的数据。需要注意的是，返回的是已经封装好的tensor，
# 且对数据进行了Data Augmentation（水平翻转、随机剪切、设置随机亮度和对比度、对数据进行标准化）
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# 再使用cifar10_input.inputs函数生成测试数据，这里不需要进行太多处理
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

# 创建数据的placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2,
                                    w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
# LRN层对ReLU会比较有用，但不适合Sigmoid这种有固定边界并且能抑制过大值的激活函数
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2,
                                    w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

# 使用一个全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 再使用一个全连接层，隐含节点数下降了一半，只有192个，其他的超参数保持不变
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层，将softmax放在了计算loss部分
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# 定义loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 获取最终的loss
loss = loss(logits, label_holder)

# 优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 使用tf.nn.in_top_k函数求输出结果中top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 使用tf.InteractiveSession创建默认的session，接着初始化全部模型参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 启动图片数据增强线程
tf.train.start_queue_runners()

# 正式开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_holder})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f'%precision)
