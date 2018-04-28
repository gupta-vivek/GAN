# -*- coding: utf-8 -*-
"""
| **@created on:** 9/4/18,
| **@author:** Vivek A Gupta,
| **@version:** v0.0.1
|
| **Description:**
| 
| **Sphinx Documentation Status:** Complete
|
..todo::
"""

import tensorflow as tf
import numpy as np
# from rztdl.contrib.tf_advance_usecases.dcgan_model import read_data
from scipy.special import expit
from PIL import Image
import datetime
from MNIST import read_data

out_shape = None
# Sample noise.
def sample_noise(batch_size, z_dim):
    return np.random.uniform(-1., 1., size=[batch_size, z_dim])


# Discriminator Network.
with tf.name_scope(name="discriminator"):
    def discriminator(x, reuse_variables=False):
        global out_shape
        alpha = 0.2
        with tf.variable_scope("discriminator", reuse=reuse_variables):
            # Reshape the image
            x = sample_noise(100, 2352)
            # x = np.array([x.reshape(28, 28, 3) for x in x])

            x = tf.reshape(x, [-1, 28, 28, 3])
            x = tf.cast(x, dtype=tf.float32)

            # Convolution Layer 1
            disc1 = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding="same")
            disc1 = tf.maximum(alpha * disc1, disc1)

            # Convolution Layer 2
            disc2 = tf.layers.conv2d(disc1, filters=128, kernel_size=5, strides=2, padding="same")
            disc2 = tf.layers.batch_normalization(disc2, training=True)
            disc2 = tf.maximum(alpha * disc2, disc2)

            # Convolution Layer 3
            disc3 = tf.layers.conv2d(disc2, filters=256, kernel_size=5, strides=1, padding="same")
            disc3 = tf.layers.batch_normalization(disc3, training=True)
            disc3 = tf.maximum(alpha * disc3, disc3)

            # out_shape = tf.shape(disc3)
            # Flatten
            # flat = tf.reshape(disc3, (-1, 4 * 4 * 256))
            flat = tf.reshape(disc3, (-1, 1254400))

            # Logits
            logits = tf.layers.dense(flat, 1)

            # Output
            out = tf.sigmoid(logits)

            return out

# Generator Network.
with tf.name_scope(name="generator"):
    def generator(z, out_channel_dim=2352, is_train=True):
        alpha = 0.2
        # with tf.variable_scope('generator', reuse=False if is_train == True else True):
        with tf.variable_scope('generator'):
            # First fully connected layer
            x_1 = tf.layers.dense(z, 2 * 2 * 512)

            # Reshape it to start the convolutional stack
            deconv_2 = tf.reshape(x_1, (-1, 2, 2, 512))
            batch_norm2 = tf.layers.batch_normalization(deconv_2, training=is_train)
            lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

            # Deconv 1
            deconv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='VALID')
            batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
            lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

            # Deconv 2
            deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding='SAME')
            batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
            lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

            # Output layer
            logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 5, 2, padding='SAME')

            out = tf.tanh(logits)

            return out

batch_size = 100
z_dim = 2352

# Training data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()
train_data = read_data.divide_batches(train_data, batch_size)
print(train_data)
print("train")
print(train_data[0])
# Place holders.
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
z = tf.placeholder(tf.float32, shape=[None, 100])

# Learning rate
learning_rate = 0.0002

disc_real = discriminator(x)  # Real MNIST Images.
gen_sample = generator(z, z_dim)  # Generated Images.
disc_fake = discriminator(gen_sample, reuse_variables=True)  # Passing generated images to discriminator.

# Get training variables.
tvars = tf.trainable_variables()
disc_vars = [var for var in tvars if var.name.startswith('discriminator')]
gen_vars = [var for var in tvars if var.name.startswith('generator')]
print("D vars - ", disc_vars)
print("G vars - ", gen_vars)

disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
disc_loss = disc_loss_fake + disc_loss_real
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))

# Optimizers.
disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=disc_vars)
gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars)

# Reuse the variables.
tf.get_variable_scope().reuse_variables()

# Initialize the variables.
init = tf.global_variables_initializer()

#Summary stastics for TensorBoard.
tf.summary.scalar('Generator_loss', gen_loss)
tf.summary.scalar('Discriminator_loss', disc_loss)

images_for_tensorboard = tf.sigmoid(generator(z, z_dim))
tf.summary.image('Generated_images', images_for_tensorboard, 5)
summary_merged = tf.summary.merge_all()
logdir = "tensorboard_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Model saver.
model_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(out_shape))

    # Summary Writer.
    writer = tf.summary.FileWriter(logdir, sess.graph)
    writer.add_graph(sess.graph)

    g_loss = None
    d_loss = None
    data = None

    # Pre train the discriminator on both real and fake images.
    print("Pre training the discriminator...")
    for i in range(1):
        for data in train_data:
            batch_z = sample_noise(batch_size, z_dim)
            _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: data, z: batch_z})

        print("Epoch - {}".format(i))
        print("Discriminator Loss - {}".format(d_loss))

    print("Pre training completed.")
    print("Generator training...")
    count = 0
    for i in range(1):
        count += 1
        for data in train_data:
            batch_z = sample_noise(batch_size, z_dim)
            _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: data, z: batch_z})
            _, g_loss = sess.run([gen_opt, gen_loss], feed_dict={z: batch_z})

        if i % 1 == 0:
            print("\nEpoch - {}".format(i))
            print("Discriminator Loss - {}".format(d_loss))
            print("Generator Loss - {}".format(g_loss))

            batch_z = sample_noise(batch_size, z_dim)
            sample_output = sess.run(gen_sample, feed_dict={z: batch_z})

            for ind, image in enumerate(sample_output):
                image = expit(image)
                image = image.reshape([28, 28]).astype('uint8')*255
                img = Image.fromarray(image)
                img.save('image/' + str(ind) + '.png')

            summary = sess.run(summary_merged, {z: batch_z, x: data})
            writer.add_summary(summary, i)
            model_saver.save(sess, 'session/gan_cnn_mnist_model', global_step=1000)


