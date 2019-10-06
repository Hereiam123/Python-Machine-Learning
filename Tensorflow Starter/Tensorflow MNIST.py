import tensorflow.compat.v1 as tf
import input_data
mnist = input_data.read_data_sets("MNIST Data/", one_hot=True)
tf.disable_v2_behavior()

print(type(mnist))

print(mnist.train.images.shape)
