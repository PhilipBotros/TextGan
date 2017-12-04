# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class LSTM_Discriminator():
    """LSTM discriminator that outputs a probability of the given sequence being real"""

    def __init__(self, Z_dim, y_dim, X_dim, vocab_size, seq_len, batch_size):

        self.y_dim = y_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        h_dim = y_dim + Z_dim
        self.batch_size = batch_size

        with tf.variable_scope("Discriminator") as scope:
            self.X = tf.placeholder(tf.int32, shape=[None, X_dim], name = "Inputs")
            self.y = tf.placeholder(tf.int32, shape=[None], name = 'Labels')

            # Initializers
            self.weight_initializer = tf.contrib.layers.xavier_initializer()
            self.const_initializer = tf.constant_initializer()
            # linear output layer of the LSTM
            self.W_out = tf.get_variable("D_W_out", shape=(
                h_dim, self.vocab_size), initializer=self.weight_initializer)
            self.b_out = tf.get_variable("D_b_out", shape=(1, self.vocab_size),
                                         initializer=self.const_initializer)

        self.LSTM = tfr.BasicLSTMCell(h_dim)
    
        self.theta_D = self._build_discriminator()


    def _build_discriminator(self):
        """Create output distribution by looping over LSTM"""
        
        x = tf.one_hot(self.X, depth=self.vocab_size)
        y = tf.one_hot(self.y, depth=self.y_dim)

        with tf.variable_scope("Discriminator") as scope:
            state = self.LSTM.zero_state(self.batch_size, dtype=tf.float32)
            for i in range(self.seq_len):

                # Concatenate input and label 
                inputs = tf.concat(x[:, i, :] + y, axis = 1)
                output, state = self.LSTM(inputs = inputs, state = state)
            
            self.logits = tf.add(tf.matmul(output, self.W_out), self.b_out, name="logits")

            # Get discrimator weights for the gradient update
            trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        return trainable_vars

    def inference(self, x, y):
        """ CAN WE DO THIS ALL WITH PLACEHOLDERS??"""

        x = tf.one_hot(x, depth=self.vocab_size)
        with tf.variable_scope("Discriminator", reuse = True) as scope:
            state = self.LSTM.zero_state(self.batch_size, dtype=tf.float32)
            for i in range(self.seq_len):
                inputs = tf.concat(x[:, i, :] + y, axis = 1)
                output, state = self.LSTM(inputs = inputs, state = state)

            logits = tf.add(tf.matmul(output, self.W_out), self.b_out, name="logits")

        return logits

    def loss(self, logit, label):
        """Discriminator loss. Input are the logits given real or fake data and a label [0,1]"""
        if label == 1:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=logit, labels=tf.ones_like(logit)))
        elif label == 0:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=logit, labels=tf.zeros_like(logit)))
