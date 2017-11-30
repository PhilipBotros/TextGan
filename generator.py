# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class Generator():

    def __init__(self, Z_dim, y_dim, h_dim):

        # inputs
        self.Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])

        # weights and biases
        self.G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
        self.G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

        # trainable variables of the generator
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self._build_generator()

    def _build_generator(self):
        inputs = tf.concat(axis=1, values=[self.Z, self.y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        self.G_prob = tf.nn.sigmoid(G_log_prob)
