import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello = tf.constant('Hello World')

# Tensor flow object
print(type(hello))

sess = tf.Session()

sess.run(hello)

# Operations

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition: ', sess.run(x+y))
    print('Subtraction: ', sess.run(x-y))
    print('Multiplication: ', sess.run(x*y))
    print('Division: ', sess.run(x/y))

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
