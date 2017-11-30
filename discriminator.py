# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class Discriminator():

    def __init__(self, y_dim, X_dim, h_dim):

        # Real data input
        self.X = tf.placeholder(tf.float32, shape=[None, 784])

        # Initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # Weights and biases
        self.D_W1 = tf.get_variable(
            'D_W1', shape=[X_dim + y_dim, h_dim], initializer=self.weight_initializer)
        self.D_b1 = tf.get_variable('D_b1', shape=[h_dim], initializer=self.const_initializer)
        self.D_W2 = tf.get_variable('D_W2', shape=[h_dim, 1], initializer=self.weight_initializer)
        self.D_b2 = tf.get_variable('D_b2', shape=[1], initializer=self.const_initializer)

        # Trainable variables of the discriminator
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def run(self, y, X):
        """Discriminator feedforward graph."""
        inputs = tf.concat(axis=1, values=[X, y])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_logit, D_prob

    def loss(self, logit, label):
        """Discriminator loss. Input are the logits given real or fake data and a label [0,1]"""
        if label == 1:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=logit, labels=tf.ones_like(logit)))
        elif label == 0:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=logit, labels=tf.zeros_like(logit)))
