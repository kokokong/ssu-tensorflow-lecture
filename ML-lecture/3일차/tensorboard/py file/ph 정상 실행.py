
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

with tf.name_scope('Image') as scope:
    filename = "cat.jpg"
    image = mpimg.imread(filename)
    height, width, color = image.shape
with tf.name_scope('transpose') as scope:
    x = tf.placeholder('uint8', [height, width, 3])
    image_input = tf.reshape(image, [-1, height, width, color])
    input_sum = tf.summary.image('input', image_input, 3)
    print(image.shape)
    transpose_x = tf.transpose(x, perm=[1, 0, 2], name='transpose')
    transpose_img = tf.reshape(transpose_x, [-1, width, height, 3])
    output_sum = tf.summary.image('output', transpose_img, 3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/Do_plt", sess.graph)
    result = sess.run(transpose_x, feed_dict={x:image})
    summary = sess.run(merged,feed_dict={x:image})
    writer.add_summary(summary)

plt.imshow(result)
plt.show()


# In[ ]:



