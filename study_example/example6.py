import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [3]])

product = tf.matmul (matrix1, matrix2)  # == np.dot(m1,m2)

#sess = tf.Session()  #create session object
#result = sess.run(product)
#print(result)
#sess.close()

with tf.Session() as sess:  # it will auto close Session object
  result2 = sess.run(product)
  print(result2)

