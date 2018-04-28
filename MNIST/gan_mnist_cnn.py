# -*- coding: utf-8 -*-
"""
@created on: 22/1/18,
@author: Vivek A Gupta,
@version: v0.0.1
Description:
Sphinx Documentation Status:
..todo::
Use leaky relu as activation function.
"""

import tensorflow as tf
from MNIST import read_data
import numpy as np
import datetime

LOSS_1 = True  # Sigmoid Cross Entropy.
LOSS_2 = False  # As mentioned in the paper.


# Sample noise.
def sample_noise(batch_size, z_dim):
    return np.random.uniform(-1., 1., size=[batch_size, z_dim])


# Discriminator network.
with tf.name_scope("discriminator"):
    def discriminator(x, reuse_variables=False):
        with tf.variable_scope("discriminator", reuse=reuse_variables):
            # Reshape the data.
            x = tf.reshape(x, shape=[-1, 28, 28, 1], name="reshape_input")

            # Weights.
            disc_weights = {
                # 5 x 5 filter, 1 input, 32 outputs.
                'wc1': tf.get_variable('d_wc1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer()),
                # 5 x 5 filter, 32 inputs, 64 outputs.
                'wc2': tf.get_variable('d_wc2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer()),
                # Fully connected, 7 x 7 x 64 inputs, 1024 inputs.
                'wf1': tf.get_variable('d_wf1', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer()),
                # 1024 inputs, 10 outputs
                'out': tf.get_variable('d_wout', [1024, 1], initializer=tf.truncated_normal_initializer())
            }

            # Biases.
            disc_biases = {
                'bc1': tf.get_variable('d_bc1', [32], initializer=tf.truncated_normal_initializer()),
                'bc2': tf.get_variable('d_bc2', [64], initializer=tf.truncated_normal_initializer()),
                'bf1': tf.get_variable('d_bf1', [1024], initializer=tf.truncated_normal_initializer()),
                'out': tf.get_variable('d_bout', [1], initializer=tf.truncated_normal_initializer())
            }

            # Layer 1.
            d1 = tf.nn.conv2d(x, disc_weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            d1 = d1 + disc_biases['bc1']
            d1 = tf.nn.max_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            d1 = tf.nn.relu(d1)

            # Layer 2.
            d2 = tf.nn.conv2d(d1, disc_weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
            d2 = d2 + disc_biases['bc2']
            d2 = tf.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            d2 = tf.nn.relu(d2)

            # Fully connected layer.
            intermediate = tf.reshape(d2, shape=[-1, 7 * 7 * 64])
            intermediate = tf.matmul(intermediate, disc_weights['wf1'])
            intermediate = tf.add(intermediate, disc_biases['bf1'])
            intermediate = tf.nn.relu(intermediate)

            # Output layer.
            out = tf.add(tf.matmul(intermediate, disc_weights['out']), disc_biases['out'])

            return out


# Generator network.
with tf.name_scope("Generator"):
    def generator(z, z_dim):
        gen_weights = {
            'w1': tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer()),
            'wc1': tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer()),
            'wc2': tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer()),
            'wc3': tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        }

        gen_biases = {
            'b1': tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer()),
            'bc1': tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer()),
            'bc2': tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer()),
            'bc3': tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer())
        }

        g1 = tf.matmul(z, gen_weights['w1']) + gen_biases['b1']
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features.
        g2 = tf.nn.conv2d(g1, gen_weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + gen_biases['bc1']
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])

        # Generate 25 features.
        g3 = tf.nn.conv2d(g2, gen_weights['wc2'], strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + gen_biases['bc2']
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])

        # Final convolution with one output channel.
        g4 = tf.nn.conv2d(g3, gen_weights['wc3'], strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + gen_biases['bc3']
        # g4 = tf.sigmoid(g4)

        # Dimensions of g4: batch_size x 28 x 28 x 1
        return g4


batch_size = 100
z_dim = 100

# Training data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()
train_data = read_data.divide_batches(train_data, batch_size)

# Place holders.
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="discriminator_placeholder")
    z = tf.placeholder(tf.float32, shape=[None, 100], name="generator_placeholder")

# Learning rate
learning_rate = 0.0002

disc_real = discriminator(x)  # Real MNIST Images.
gen_sample = generator(z, z_dim)  # Generated Images.
disc_fake = discriminator(gen_sample, reuse_variables=True)  # Passing generated images to discriminator.

# Get training variables.
tvars = tf.trainable_variables()
disc_vars = [var for var in tvars if 'd_' in var.name]
gen_vars = [var for var in tvars if 'g_' in var.name]

# Loss.
disc_loss = None
gen_loss = None

with tf.name_scope("loss"):
    if LOSS_1:
        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)), name="disc_fake_loss")
        disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)), name="disc_real_loss")
        disc_loss = disc_loss_fake + disc_loss_real
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)), name="gen_loss")

    if LOSS_2:
        disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1.-disc_fake), name="disc_loss")
        gen_loss = -tf.reduce_mean(tf.log(gen_sample), name="gen_loss")

# Optimizers.
with tf.name_scope("Optimizers"):
    disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=disc_vars, name="disc_optimizer")
    gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars, name="gen_optimizer")

# Reuse the variables.
tf.get_variable_scope().reuse_variables()

# Initialize the variables.
init = tf.global_variables_initializer()

#Summary stastics for TensorBoard.
with tf.name_scope("summary"):
    tf.summary.scalar('Generator_loss', gen_loss)
    tf.summary.scalar('Discriminator_loss', disc_loss)

    images_for_tensorboard = generator(z, z_dim)
    tf.summary.image('Generated_images', images_for_tensorboard, 5)
    summary_merged = tf.summary.merge_all()

logdir = "tensorboard_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

    # Pre train the discriminator on both real and fake images. This is optional.
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

            summary = sess.run(summary_merged, {z: batch_z, x: data})
            writer.add_summary(summary, i)
            model_saver.save(sess, 'session/gan_cnn_mnist_model', global_step=1000)

            # Saving the images.
            # for ind, image in enumerate(sample_output):
            #     image = expit(image)
            #     image = image.reshape([28, 28]).astype('uint8')*255
            #     img = Image.fromarray(image)
            #     img.save('image/' + str(ind) + '.png')

