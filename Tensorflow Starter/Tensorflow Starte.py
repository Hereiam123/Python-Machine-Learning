import tensforflow as tf
hello = tf.constant('Hello World')

# Tensor flow object
print(type(hello))

sess = tf.Session()

sess.run(hello)
