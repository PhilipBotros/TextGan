# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class Discriminator():

    def __init__(self, y_dim, X_dim, h_dim):

        # inputs
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])

        # weights
        self.D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
        self.D_W2 = tf.Variable(xavier_init([h_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        # trainable variables of the discriminator
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def _build_discriminator(self):
        inputs = tf.concat(axis=1, values=[x, y])
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit
