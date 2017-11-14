from __future__ import print_function

import tensorflow as tf
import numpy as np

def get_mask(in_channels):
    mask = np.zeros((3, 3, in_channels, in_channels))
    for _ in range(in_channels):
        mask[:, :, _, _] = 1.
    return mask

def Conv3x3(x, in_channels, out_channels, stride=1, is_training=True,
        scope='convolution'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weight', shape=(3, 3, in_channels, out_channels),
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, w, (1, 1, stride, stride), 'SAME',
                         data_format='NCHW')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         data_format='NCHW', fused=True,
                                         is_training=is_training)
        x = tf.nn.relu(x)
        return x

def DiagonalwiseRefactorization(x, in_channels, stride=1, groups=4,
        is_training=True, scope='depthwise'):
    with tf.variable_scope(scope):
        channels = in_channels / groups
        mask = tf.constant(get_mask(channels).tolist(), dtype=tf.float32,
                           shape=(3, 3, channels, channels))
        splitw = [
            tf.get_variable('weights_%d' % _, (3, 3, channels, channels),
                            initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(groups)
        ]
        splitw = [tf.multiply(w, mask) for w in splitw]
        splitx = tf.split(x, groups, 1)
        splitx = [tf.nn.conv2d(x, w, (1, 1, stride, stride), 'SAME',
                               data_format='NCHW')
                  for x, w in zip(splitx, splitw)]
        x = tf.concat(splitx, 1)
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         data_format='NCHW', fused=True,
                                         is_training=is_training)
        x = tf.nn.relu(x)
        return x

def Depthwise(x, in_channels, stride=1, is_training=True, scope='depthwise'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (3, 3, in_channels, 1),
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.depthwise_conv2d(x, w, (1, 1, stride, stride), 'SAME',
                                   data_format='NCHW')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         data_format='NCHW', fused=True,
                                         is_training=is_training)
        x = tf.nn.relu(x)
        return x

def DiagonalwiseRefactorizationWithoutGrouping(x, in_channels, stride=1, is_training=True,
        scope='depthwise'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (3, 3, in_channels, in_channels),
                            initializer=tf.contrib.layers.xavier_initializer())
        mask = tf.constant(get_mask(in_channels).tolist(), dtype=tf.float32,
                           shape=(3, 3, in_channels, in_channels))
        w = tf.multiply(w, mask)
        x = tf.nn.conv2d(x, w, (1, 1, stride, stride), 'SAME',
                         data_format='NCHW')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         data_format='NCHW', fused=True,
                                         is_training=is_training)
        x = tf.nn.relu(x)
        return x

def Pointwise(x, in_channels, out_channels, is_training=True,
        scope='pointwise'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (1, 1, in_channels, out_channels),
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, w, (1, 1, 1, 1), 'SAME', data_format='NCHW')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         data_format='NCHW', fused=True,
                                         is_training=is_training)
        x = tf.nn.relu(x)
        return x

def Separable(x, in_channels, out_channels, stride=1, is_training=True,
        scope='separable'):
    with tf.variable_scope(scope):
        # Diagonalwise Refactorization
        # groups = in_channels
        # groups = 16
        groups = max(in_channels / 32, 1)
        x = DiagonalwiseRefactorization(x, in_channels, stride, groups,
                                        is_training, 'depthwise')

        # Specialized Kernel
        # x = Depthwise(x, in_channels, stride, is_training, 'depthwise')

        # Standard Convolution
        # x = Conv3x3(x, in_channels, in_channels, stride, is_training,
                    # 'convolution')
        x = Pointwise(x, in_channels, out_channels, is_training, 'pointwise')
        return x

def AvgPool7x7(x):
    x = tf.nn.avg_pool(x, (1, 1, 7, 7), (1, 1, 1, 1), 'VALID',
                       data_format='NCHW')
    x = tf.squeeze(x, axis=(2, 3))
    return x

def Linear(x, in_channels, out_channels, scope='fc'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', (in_channels, out_channels),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases', out_channels,
                            initializer=tf.constant_initializer(0.))
        x = tf.matmul(x, w) + b
        return x
