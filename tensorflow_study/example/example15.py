import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
  layer_name = 'layer%s' % n_layer
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
      tf.histogram_summary(layer_name + '/weights', Weights)
#      tf.scalar_summary(layer_name+'/weights', Weights)
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name = 'b')
      tf.histogram_summary(layer_name + '/biases', biases)
#      tf.scalar_summary(layer_name+'/biases', biases)
    with tf.name_scope('Wx_plus_b'):
      Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
      outputs = Wx_plus_b
    else:
      outputs = activation_function(Wx_plus_b)
    tf.histogram_summary(layer_name + '/outputs', outputs)
#    tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs

#make data
x_data = np.linspace (-1,1,300)[:,np.newaxis]  # value is -1~1, 300 row
noise = np.random.normal(0,0.05,x_data.shape) # mean = 0, std = 0.05
y_data = np.square(x_data) - 0.5 + noise

# placeholder used to implement batch training
with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
  ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

# asuume 3 layer
# 1 => 10 => 1

L1 = add_layer(xs, 1, 10, 1, tf.nn.relu)
prediction = add_layer(L1, 10, 1, 2,None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                        reduction_indices=[1]))
  tf.scalar_summary('loss', loss)

with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # learning rate < 1


sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())


for  i in range(10000):
  sess.run(train_step, feed_dict={xs: x_data, ys:y_data})
  if i%50 == 0:
    result = sess.run(merged, feed_dict={xs: x_data, ys:y_data})
    writer.add_summary(result , i)

