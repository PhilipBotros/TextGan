# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class Discriminator():

    def __init__(self, y_dim, X_dim, h_dim):

        # inputs
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # Weights and biases
        self.D_W1 = tf.get_variable('D_W1', shape = [X_dim + y_dim, h_dim], initializer = self.weight_initializer)
        self.D_b1 = tf.get_variable('D_b1', shape = [h_dim], initializer = self.const_initializer)
        self.D_W2 = tf.get_variable('D_W2', shape = [h_dim, 1], initializer = self.weight_initializer)
        self.D_b2 = tf.get_variable('D_b2', shape = [1], initializer = self.const_initializer)

        # trainable variables of the discriminator
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
        self._build_discriminator()

    def _build_discriminator(self):
        inputs = tf.concat(axis=1, values=[self.X, self.y])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        self.D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        self.D_prob = tf.nn.sigmoid(self.D_logit)

    def _loss(self):
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit, labels=self.labels)