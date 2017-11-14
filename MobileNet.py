from __future__ import print_function

import tensorflow as tf
from ops import Separable, Conv3x3, AvgPool7x7, Linear

def MobileNet(x, is_training=True):
    with tf.variable_scope('mobilenet'):
        x = Conv3x3(x, 3, 32, 2, is_training, 'conv1') # 224 -> 112
        x = Separable(x, 32, 64, 1, is_training, 'sepa1')
        x = Separable(x, 64, 128, 2, is_training,'sepa2') # 112 -> 56
        x = Separable(x, 128, 128, 1, is_training, 'sepa3_1')
        x = Separable(x, 128, 256, 2, is_training, 'sepa3_2') # 56 -> 28
        x = Separable(x, 256, 256, 1, is_training, 'sepa4_1')
        x = Separable(x, 256, 512, 2, is_training, 'sepa4_2') # 28 -> 14
        for i in range(5):
            x = Separable(x, 512, 512, 1, is_training, 'sepa5_{}'.format(i + 1))
        x = Separable(x, 512, 1024, 2, is_training, 'sepa5_6') # 14 -> 7
        x = Separable(x, 1024, 1024, 1, is_training, 'sepa6')
        x = AvgPool7x7(x)
        x = Linear(x, 1024, 1000)
        return x

