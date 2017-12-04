# Tensorflow modules
import tensorflow as tf


class Generator():
    """A simple MLP generator"""

    def __init__(self, Z_dim, y_dim, h_dim, X_dim):

        # Initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # Weights and biases
        self.G_W1 = tf.get_variable(
            'G_W1', shape=[Z_dim + y_dim, h_dim], initializer=self.weight_initializer)
        self.G_b1 = tf.get_variable('G_b1', shape=[h_dim], initializer=self.const_initializer)
        self.G_W2 = tf.get_variable(
            'G_W2', shape=[h_dim, X_dim], initializer=self.weight_initializer)
        self.G_b2 = tf.get_variable('G_b2', shape=[X_dim], initializer=self.const_initializer)

        # Trainable variables of the generator
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # Build computational graph
        self._build_generator()

    def generate(self, y, z):
        """Generator feedforward graph"""
        inputs = tf.concat(axis=1, values=[self.Z, self.y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        self.G_prob = tf.nn.sigmoid(G_log_prob)

        

