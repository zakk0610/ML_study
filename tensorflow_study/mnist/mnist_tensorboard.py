import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def add_layer(inputs, in_size, out_size, layer_name, activation_function = None):
  Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
  #Weights = tf.Variable(tf.random_normal([in_size, out_size]))  #failed
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  Wx_plus_b = tf.matmul(inputs, Weights) + biases
  if activation_function is None:
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  tf.histogram_summary(layer_name + '/outputs', outputs)
  return outputs

# argmax: Returns the index with the largest value across dimensions of a tensor.
def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs})
  correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
  return result

xs = tf.placeholder(tf.float32, [None, 784])  # 28 x 28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)

L1 = add_layer(xs, 784, 100, 'L1', activation_function= tf.nn.relu)
dropouted = tf.nn.dropout(L1, keep_prob)
prediction =add_layer(dropouted, 100, 10, 'L2', activation_function= tf.nn.softmax)

#cross_entropy = -tf.reduce_mean(ys * tf.log(prediction))  #loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.scalar_summary('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()
train_writer=tf.train.SummaryWriter("logs/train",sess.graph)
test_writer=tf.train.SummaryWriter("logs/test",sess.graph)

sess.run(tf.initialize_all_variables())

for i in range (500):
  sess.run(train_step, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, keep_prob:0.5})
  if i % 50 == 0:
#    print (compute_accuracy(mnist.test.images, mnist.test.labels))
    train_result = sess.run(merged,feed_dict= {xs: mnist.train.images, ys: mnist.train.labels, keep_prob:1})
    train_writer.add_summary(train_result,i)
    test_result = sess.run(merged, feed_dict = {xs: mnist.test.images, ys: mnist.test.labels, keep_prob:1})
    test_writer.add_summary(test_result,i)

