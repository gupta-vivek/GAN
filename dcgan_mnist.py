# -*- coding: utf-8 -*-
"""
@created on: 22/1/18,
@author: Vivek A Gupta,
@version: v0.0.1
Description:
Sphinx Documentation Status:
..todo::
"""

import tensorflow as tf
import read_data
import numpy as np
import datetime

with tf.name_scope("lrelu"):
    def lrelu(x, th=0.2):
        return tf.maximum(th * x, x)


# Sample noise.
def sample_noise(batch_size, z_dim):
    return np.random.uniform(-1., 1., size=[batch_size, z_dim])


# Discriminator network.
def discriminator(x, reuse_variables=False):
        with tf.variable_scope("discriminator", reuse=reuse_variables):
            x = tf.reshape(x, shape=[-1, 28, 28, 1], name="reshape_input")
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=lrelu, name="conv_1")
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=lrelu, name="conv_2")
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=lrelu, name="conv_3")
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=128, activation=lrelu, name="ffn")
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid, name="output_layer")
            return x


# Generator network.
def generator(z):
    activation = lrelu
    momentum = 0.99
    is_training = True
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation, name="ffn_1")
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.reshape(x, shape=[-1, d1, d1, d2], name="reshape_ffn_1")
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation, name="deconv_1")
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation, name="deconv_2")
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation, name="deconv_3")
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.tanh, name="output_layer")
        return x


batch_size = 100
z_dim = 64

# Training data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()
train_data = read_data.divide_batches(train_data, batch_size)

# Place holders.
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="discriminator_placeholder")
    z = tf.placeholder(tf.float32, shape=[None, 64], name="generator_placeholder")

# Learning rate
learning_rate = 0.0002

disc_real = discriminator(x)  # Real MNIST Images.
gen_sample = generator(z)  # Generated Images.
disc_fake = discriminator(gen_sample, reuse_variables=True)  # Passing generated images to discriminator.


disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

with tf.name_scope("loss"):
    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)), name="disc_loss_real")
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake), name="disc_loss_fake"))
    disc_loss = disc_loss_fake + disc_loss_real
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)), name="gen_loss")

# Optimizers.
with tf.name_scope("optimizers"):
    disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=disc_vars, name="disc_optimizer")
    gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars, name="gen_optimizer")

# Reuse the variables.
tf.get_variable_scope().reuse_variables()

# Initialize the variables.
init = tf.global_variables_initializer()

#Summary stastics for TensorBoard.
tf.summary.scalar('Generator_loss', gen_loss)
tf.summary.scalar('Discriminator_loss', disc_loss)

images_for_tensorboard = generator(z)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
summary_merged = tf.summary.merge_all()
logdir = "tensorboard_dcgan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Model saver.
model_saver = tf.train.Saver()
with tf.device("/device:GPU:0"):
    with tf.Session() as sess:
        sess.run(init)

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
                summary = sess.run(summary_merged, {z: batch_z, x: data})
                writer.add_summary(summary, i)
                model_saver.save(sess, 'session/dcgan_cnn_mnist_model', global_step=20)
