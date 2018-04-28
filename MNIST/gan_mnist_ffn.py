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
from MNIST import read_data
import numpy as np
from PIL import Image
import datetime
from scipy.special import expit

LOSS_1 = True  # Sigmoid Cross Entropy.
LOSS_2 = False  # As mentioned in the paper.


# Sample noise.
def sample_noise(batch_size, z_dim):
    return np.random.uniform(-1., 1., size=[batch_size, z_dim])


# Discriminator network.
def discriminator(x, reuse_variables=False):
    with tf.variable_scope("discriminator", reuse=reuse_variables):

        # Weights.
        disc_weights = {
            'w1': tf.get_variable('d_w1', [784, 128], initializer=tf.truncated_normal_initializer()),
            'wout': tf.get_variable('d_wout', [128, 1], initializer=tf.truncated_normal_initializer())
        }

        # Biases.
        disc_biases = {
            'b1': tf.get_variable('d_b1', [128], initializer=tf.truncated_normal_initializer()),
            'bout': tf.get_variable('d_bout', [1], initializer=tf.truncated_normal_initializer())
        }

        disc_hidden = tf.nn.relu(tf.add(tf.matmul(x, disc_weights['w1']), disc_biases['b1']))
        disc_out = tf.add(tf.matmul(disc_hidden, disc_weights['wout']), disc_biases['bout'])

        return disc_out


# Generator network.
def generator(z):
    gen_weights = {
        'w1': tf.get_variable('g_w1', [100, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer()),
        'wout': tf.get_variable('g_wout', [128, 784], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
    }

    gen_biases = {
        'b1': tf.get_variable('g_b1', [128], initializer=tf.truncated_normal_initializer()),
        'bout': tf.get_variable('g_bout', [784], initializer=tf.truncated_normal_initializer())
    }

    gen_hidden = tf.add(tf.matmul(z, gen_weights['w1']), gen_biases['b1'])
    gen_out = tf.add(tf.matmul(gen_hidden, gen_weights['wout']), gen_biases['bout'])

    return gen_out



batch_size = 100
z_dim = 100

# Training data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()
train_data = read_data.divide_batches(train_data, batch_size)

# Place holders.
x = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

# Learning rate
learning_rate = 0.0002

disc_real = discriminator(x)  # Real MNIST Images.
gen_sample = generator(z)  # Generated Images.
disc_fake = discriminator(gen_sample, reuse_variables=True)  # Passing generated images to discriminator.

# Get training variables.
tvars = tf.trainable_variables()
disc_vars = [var for var in tvars if 'd_' in var.name]
gen_vars = [var for var in tvars if 'g_' in var.name]

# Loss.
disc_loss = None
gen_loss = None

if LOSS_1:
    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_fake + disc_loss_real
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))

if LOSS_2:
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1.-disc_fake))
    gen_loss = -tf.reduce_mean(tf.log(gen_sample))

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

images_for_tensorboard = tf.sigmoid(tf.reshape(generator(z), [-1, 28, 28, 1]))
tf.summary.image('Generated_images', images_for_tensorboard, 5)
summary_merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Model saver.
model_saver = tf.train.Saver()

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
    count = 0
    for i in range(200):
        count += 1
        for data in train_data:
            batch_z = sample_noise(batch_size, z_dim)
            _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: data, z: batch_z})

        if count % 10 == 0:
            print("Epoch - {}".format(i))
            print("Discriminator Loss - {}".format(d_loss))

    print("Pre training completed.")
    print("Generator training...")
    count = 0
    for i in range(3000):
        count += 1
        for data in train_data:
            batch_z = sample_noise(batch_size, z_dim)
            _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: data, z: batch_z})
            _, g_loss = sess.run([gen_opt, gen_loss], feed_dict={z: batch_z})

        if i % 10 == 0:
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
            model_saver.save(sess, 'session/gan_mnist_model', global_step=1000)