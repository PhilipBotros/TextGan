import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
from discriminator import Discriminator
from generator import Generator
from lstm_generator import LSTM_Generator
from auxiliary import sample_Z, plot


# TODO: read sentences, not images
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

# TODO:  add LSTM constants
# TODO: command line args perhaps?
BATCH_SIZE = 64
Z_DIM = 100
X_DIM = mnist.train.images.shape[1]
COND_DIM = mnist.train.labels.shape[1]
HIDDEN_DIM = 128
N_SAMPLE = 16

#-- Build Graph ------------------------------------------------------------------------------------
# Intialize generator and discriminator
# TODO: take placeholders outside of classes, it stopped making any sense
generator = Generator(Z_DIM, COND_DIM, HIDDEN_DIM, X_DIM)
discriminator = Discriminator(COND_DIM, X_DIM, HIDDEN_DIM)
lstm_generator = LSTM_Generator(Z_DIM, COND_DIM, X_DIM, 100, 30, BATCH_SIZE)

# Add discriminator outputs
D_real, D_logit_real = discriminator.run(generator.y, discriminator.X)
D_fake, D_logit_fake = discriminator.run(generator.y, generator.G_prob)

# Add discriminator and generator loss
D_loss = discriminator.loss(D_real, 1) + discriminator.loss(D_fake, 0)
G_loss = discriminator.loss(D_fake, 1)

# Add optimizers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator.theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator.theta_G)

#-- Test LSTM generator ----------------------------------------------------------------------------

#-- Train ------------------------------------------------------------------------------------------
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    #-- Test LSTM generator ------------------------------------------------------------------------
    Z_sample = sample_Z(N_SAMPLE, Z_DIM)
    y_sample = np.zeros(shape=[N_SAMPLE, COND_DIM])
    samples = sess.run(lstm_generator.samples, feed_dict={
                       lstm_generator.y: y_sample, lstm_generator.Z: Z_sample})
    print("LSTM generator random samples")
    print(samples)
    #-----------------------------------------------------------------------------------------------

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

#-- The End ----------------------------------------------------------------------------------------
