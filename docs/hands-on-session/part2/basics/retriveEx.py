import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [2,2], name = 'x')

w1 = tf.Variable(tf.random_uniform([2,3]), name='w1')
w2 = tf.Variable(tf.random_uniform([3,1]), name='w1')
 
z1 = tf.matmul(x,w1)	# (2,3)
z2 = tf.matmul(z1,w2)	# (2,1)

with tf.Session() as sess :	
	saver = tf.train.Saver()
	saver.restore(sess, "model/my-model.ckpt")	 	
	for i in range(10):
		print sess.run( z2, {x:np.random.rand(2,2)} )



