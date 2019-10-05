import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

#hello = tf.constant('Hello World')

# Tensor flow object
# print(type(hello))

#sess = tf.Session()

# sess.run(hello)

# Operations

"""x = tf.constant(2)
y = tf.constant(3)"""

"""with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition: ', sess.run(x+y))
    print('Subtraction: ', sess.run(x-y))
    print('Multiplication: ', sess.run(x*y))
    print('Division: ', sess.run(x/y))"""

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
sub = tf.subtract(x, y)
mul = tf.multiply(x, y)
d = {x: 20, y: 30}
with tf.Session() as sess:
    print('Operations with Placeholders')
    print('addition', sess.run(add, feed_dict=d))
    print('subtraction', sess.run(sub, feed_dict=d))
    print('multiplication', sess.run(mul, feed_dict=d))

a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])

matrix1 = tf.constant(a)
matrix2 = tf.constant(b)

matrix_multi = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)
