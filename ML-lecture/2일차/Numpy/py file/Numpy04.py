
# coding: utf-8

# In[6]:

import tensorflow as tf
import numpy as np

seed = np.random.seed()
x = tf.random_uniform([5],minval=0.0,maxval=10.0,
                     dtype=tf.float32, seed=seed)
mean = tf.reduce_mean(x)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        sess.run(x)
        sess.run(mean)
        print('x: ', sess.run(x),
              'mean: ', sess.run(mean))


# In[ ]:



