import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
from discriminator import Discriminator
from generator import Generator
from auxiliary import sample_Z, plot


# TODO: read sentences, not images
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


# TODO:  add LSTM constants
BATCH_SIZE = 64
Z_DIM = 100
X_DIM = mnist.train.images.shape[1]
COND_DIM = mnist.train.labels.shape[1]
HIDDEN_DIM = 128
N_SAMPLE = 16


# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)


# """ Discriminator Net model """
# X = tf.placeholder(tf.float32, shape=[None, 784])
# y = tf.placeholder(tf.float32, shape=[None, y_dim])

# D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
# D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

# D_W2 = tf.Variable(xavier_init([h_dim, 1]))
# D_b2 = tf.Variable(tf.zeros(shape=[1]))

# theta_D = [D_W1, D_W2, D_b1, D_b2]


# def discriminator(x, y):
#     inputs = tf.concat(axis=1, values=[x, y])
#     D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
#     D_logit = tf.matmul(D_h1, D_W2) + D_b2
#     D_prob = tf.nn.sigmoid(D_logit)

#     return D_prob, D_logit


# """ Generator Net model """
# Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

# G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
# G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

# G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
# G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

# theta_G = [G_W1, G_W2, G_b1, G_b2]


# def generator(z, y):
#     inputs = tf.concat(axis=1, values=[z, y])
#     G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
#     G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#     G_prob = tf.nn.sigmoid(G_log_prob)

#     return G_prob


# def sample_Z(m, n):
#     return np.random.uniform(-1., 1., size=[m, n])
#
#
# def plot(samples):
#     fig = plt.figure(figsize=(4, 4))
#     gs = gridspec.GridSpec(4, 4)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#     return fig

# Intialize generator and discriminator
# TODO: take placeholders outside of classes, it stopped making any sense
generator = Generator(Z_DIM, COND_DIM, HIDDEN_DIM, X_DIM)
discriminator = Discriminator(COND_DIM, X_DIM, HIDDEN_DIM)

# Add discriminator outputs
D_real, D_logit_real = discriminator.run(generator.y, discriminator.X)
D_fake, D_logit_fake = discriminator.run(generator.y, generator.G_prob)

# Add discriminator and generator loss
D_loss = discriminator.loss(D_real, 1) + discriminator.loss(D_fake, 0)
G_loss = discriminator.loss(D_fake, 1)

# Add optimizers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator.theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator.theta_G)


with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Store MNIST
    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    # Joint discriminator/generator training
    for it in range(1000000):
        if it % 1000 == 0:
            # Sample generator inputs; conditional label is the seventh digit
            Z_sample = sample_Z(N_SAMPLE, Z_DIM)
            y_sample = np.zeros(shape=[N_SAMPLE, COND_DIM])
            y_sample[:, 7] = 1

            samples = sess.run(generator.G_prob, feed_dict={
                               generator.Z: Z_sample, generator.y: y_sample})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, y_mb = mnist.train.next_batch(BATCH_SIZE)

        # Joint Optimization
        Z_sample = sample_Z(BATCH_SIZE, Z_DIM)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                                  discriminator.X: X_mb, generator.Z: Z_sample, generator.y: y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                  generator.Z: Z_sample, generator.y: y_mb})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
