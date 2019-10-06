import tensorflow.compat.v1 as tf
import input_data
import matplotlib.pyplot as plt

# One hot allows comparison of true label, for one value
mnist = input_data.read_data_sets("MNIST Data/", one_hot=True)
tf.disable_v2_behavior()

# print(type(mnist))

# print(mnist.train.images.shape)

# Reshape string array for image and plot, grey scaled
# plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray')
# plt.imshow(mnist.train.images[1].reshape(
#    784, 1), cmap='gist_gray', aspect=0.02)
# plt.show()

# Batch images, in shape of 784 pixels
x = tf.placeholder(tf.float32, shape=[None, 784])

# Weights, 784 is num pixels, and 10 possible values for each image value
W = tf.Variable(tf.zeros([784, 10]))

# Bias
b = tf.Variable(tf.zeros([10]))

# Prediciton labels
y = tf.matmul(x, W) + b

# Loss and optimizer, true labels
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# Cross entropy, reduce error between the true labels and prediction images
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        # Number of batches to run on
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    print(sess.run(acc, feed_dict={
          x: mnist.test.images, y_true: mnist.test.labels}))
