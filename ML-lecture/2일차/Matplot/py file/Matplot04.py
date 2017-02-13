
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "도깨비.jpg"
image = mpimg.imread(filename)
height,width,color = image.shape

x = tf.Variable(image,name = 'x')
reverse_x = tf.reverse_sequence(x,[width]*height,1,0)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run(reverse_x)
print(result.shape)
plt.imshow(result)
plt.xticks([]),plt.yticks([])
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



