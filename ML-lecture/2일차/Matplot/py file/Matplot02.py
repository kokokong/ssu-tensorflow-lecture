
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "도깨비.jpg"
image = mpimg.imread(filename)
print("Original size:" ,image.shape)

x = tf.Variable(image,name = 'x')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    x = tf.transpose(x,perm=[1,0,2])
    sess.run(init)
    result = sess.run(x)
print("changed size: ", result.shape)
plt.imshow(result)
plt.xticks([]),plt.yticks([])
plt.show()


# In[ ]:



