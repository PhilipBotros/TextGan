import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import utils
from discriminator import Discriminator
from generator import Generator
from lstm_generator import LSTM_Generator
from lstm_discriminator import LSTM_Discriminator
from auxiliary import sample_Z, plot


# TODO:  add LSTM constants
# TODO: command line args perhaps?
BATCH_SIZE = 10
SEQ_LEN = 20
Z_DIM = 10
X_DIM = SEQ_LEN
Y_DIM = 10
HIDDEN_DIM = 10
N_SAMPLE = 16
VOCAB_SIZE = 10

#-- Build Graph ------------------------------------------------------------------------------------
# Intialize generator and discriminator
# TODO: take placeholders outside of classes, it stopped making any sense
# TODO: fix generator loss
print("Initializing GAN")
discriminator = LSTM_Discriminator(Z_DIM, Y_DIM, X_DIM, VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)
generator = LSTM_Generator(Z_DIM, Y_DIM, X_DIM, VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)

print("Gathering samples")
samples = generator.samples
c, h = generator.c, generator.h

D_logit_real = discriminator.logits
discriminator.X = samples
D_logit_fake = discriminator.logits

print("Building loss")
# Add discriminator and generator loss
D_loss = discriminator.loss(D_logit_real, 1) + discriminator.loss(D_logit_fake, 0)
# MODIFY LOSS
G_loss = generator.loss(samples, c, h, discriminator)

print("Building optimizers")
# Add optimizers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator.theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator.theta_G)

#-- Train ------------------------------------------------------------------------------------------
print("Starting session....")
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    start = -np.ones(shape=[N_SAMPLE]).astype(int)

    # Joint discriminator/generator training
    for it in range(1000000):
        if it % 1 == 0:
            # Sample generator inputs; conditional label is the seventh digit
            Z_sample = sample_Z(N_SAMPLE, Z_DIM)
            y_sample = np.zeros(shape=[N_SAMPLE, Y_DIM]).astype(float)
            y_sample[:, 7] = 1

            tr_samples = sess.run(generator.samples, feed_dict={
                generator.Z: Z_sample, generator.y: y_sample, "start_token:0": start})

            print(tr_samples)

        X_mb = utils.generate_palindrome_batch(BATCH_SIZE, SEQ_LEN)
        y_mb = X_mb[:, -1]
        # Joint Optimization
        Z_sample = sample_Z(BATCH_SIZE, Z_DIM)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                                  'Discriminator/Inputs:0': X_mb, Z: Z_sample, discriminator.y: y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                  'Discriminator/Inputs:0': X_mb, Z: Z_sample, y: y_mb})

        if it % 10 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))

#-- The End ----------------------------------------------------------------------------------------
