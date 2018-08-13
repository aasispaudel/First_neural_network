
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# In[3]:


mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500

w1 = tf.Variable(tf.random_normal([784, nodes_hl1]), dtype=tf.float32, name='weights')
b1 = tf.Variable(tf.random_normal([nodes_hl1]), dtype=tf.float32, name='biass')

w2 = tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2]), dtype=tf.float32, name='weights2')
b2 = tf.Variable(tf.random_normal([nodes_hl2]), dtype=tf.float32, name='biass2')

w3 = tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3]), dtype=tf.float32, name='weights3')
b3 = tf.Variable(tf.random_normal([nodes_hl3]), dtype=tf.float32, name='biass3')

w_f = tf.Variable(tf.random_normal([nodes_hl3, 10]), dtype=tf.float32, name='final_weights')
b_f = tf.Variable(tf.random_normal([10]), dtype=tf.float32, name='final_biass')

x = tf.placeholder(dtype=tf.float32, name='x')
y = tf.placeholder(dtype=tf.float32, name='y')

layer_1 = tf.nn.relu(tf.matmul(x, w1) + b1)
layer_2 = tf.nn.relu(tf.matmul(layer_1, w2) + b2)
layer_3 = tf.nn.relu(tf.matmul(layer_2, w3) + b3)
y_pred = tf.matmul(layer_3, w_f)


# print(tf.Session().run(y_pred, {x:x_train, y:y_train}))


# In[4]:


rmse = tf.sqrt(
                tf.reduce_mean(tf.square(y-y_pred)), name='rmse'
)

test_score = tf.divide(rmse, tf.abs(tf.reduce_mean(y)), name='nrmse')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


# In[5]:


#tf.reset_default_graph()


# In[6]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[7]:


optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())


batch_size = 100

for epoch in range(10):
    for i in range(int(mnist.train.num_examples/batch_size)):
        x_train, y_train = mnist.train.next_batch(batch_size)
        
        x_t = x_train.astype('float32')
        y_t = y_train.astype('float32')
        
        sess.run(train, feed_dict={x:x_t, y:y_t})

    print('Epoch: ', epoch)
print(r'Epochs coompleted/// data ready for test')
print(sess.run(y_pred, {x:x_t, y:y_t}))


# In[8]:


correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy:',accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

