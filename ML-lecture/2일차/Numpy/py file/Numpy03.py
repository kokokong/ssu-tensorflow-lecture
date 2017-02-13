
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

import numpy as np


# In[3]:

x= np.loadtxt('data.txt',dtype=np.float32)


# In[4]:

era = tf.Variable(x)


# In[5]:

mean = tf.reduce_mean(era)


# In[6]:

init = tf.global_variables_initializer()


# In[7]:

with tf.Session() as sess:
    sess.run(init)
    print("상위 10명의 평균자책점: ", sess.run(mean))


# In[ ]:



