import tensorflow as tf
import random
import numpy as np
from MobileNet import MobileNet
from time import clock

# preload data to eliminite data loading time
images = tf.placeholder(tf.float32, shape=[None, 3, 224, 224])
labels = tf.placeholder(tf.float32, shape=[None, 1000])
preds = MobileNet(images, True)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds)
train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

batch_size = 64
max_batches = 50

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Started.')

tot_time = 0
for _ in range(max_batches + 1):
    images_ = 2. * (np.random.rand(batch_size, 3, 224, 224) - 0.5)
    index = [random.randrange(1000) for i in range(batch_size)]
    labels_ = np.zeros(shape=(batch_size, 1000))
    labels_[range(batch_size), index] = 1.
    t0 = clock()
    sess.run(train_step, feed_dict={images: images_, labels: labels_})
    t1 = clock()
    if _ > 0:
        tot_time += t1 - t0
    print('Iter {}: {}'.format(_, t1 - t0))

avg_time = tot_time / float(max_batches)
print('{}'.format(avg_time))
