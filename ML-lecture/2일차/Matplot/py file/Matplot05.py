
# coding: utf-8

# In[ ]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "도깨비.jpg"
image = mpimg.imread(filename)
height, width, color = image.shape

x = tf.placeholder("uint8", [None, None, 3],name = 'x')
reverse_x = tf.reverse_sequence(x,[width] * height, 1,0)
reverse_y = tf.reverse_sequence(x,width * [height], 0,1)

def show_img(result,name):
    print("changed_shape: ", result.shape)
    plt.imshow(result)
    plt.xticks([]),plt.yticks([])
    plt.title(name)
    plt.show()


with tf.Session() as sess:
    result = sess.run(reverse_y,feed_dict={x:image})
    show_img(result,"top bottom")


# In[ ]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "도깨비.jpg"
image = mpimg.imread(filename)
height, width, color = image.shape

x = tf.placeholder("uint8", [None, None, 3],name = 'x')
reverse_x = tf.reverse_sequence(x,[width] * height, 1,0)
reverse_y = tf.reverse_sequence(x,width * [height], 0,1)

def show_img(result,name):
    print("changed_shape: ", result.shape)
    plt.imshow(result)
    plt.xticks([]),plt.yticks([])
    plt.title(name)
    plt.show()


with tf.Session() as sess:
    result = sess.run(reverse_y,feed_dict={x:image})
    show_img(result,"top bottom")
    result = sess.run(reverse_x, feed_dict={x: image})
    show_img(result,"left right")


# In[ ]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "도깨비.jpg"
image = mpimg.imread(filename)
height, width, color = image.shape

x = tf.placeholder("uint8", [None, None, 3],name = 'x')
reverse_x = tf.reverse_sequence(x,[width] * height, 1,0)
reverse_y = tf.reverse_sequence(x,width * [height], 0,1)

def show_img(result,name):
    print("changed_shape: ", result.shape)
    plt.imshow(result)
    plt.xticks([]),plt.yticks([])
    plt.title(name)
    plt.show()


with tf.Session() as sess:
    result = sess.run(reverse_y,feed_dict={x:image})
    show_img(result,"top bottom")
    result = sess.run(reverse_x, feed_dict={x: image})
    show_img(result,"left right")
    result = sess.run(reverse_y,feed_dict={x:result})
    show_img(result,"x y")


# In[ ]:



