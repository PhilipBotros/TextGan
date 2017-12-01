# Tensorflow modules
import tensorflow as tf
tfr = tf.contrib.rnn
tfg = tf.contrib.rnn


class LSTM_Generator():
    """LSTM generator that has text io"""

    def __init__(self, Z_dim, y_dim, X_dim, vocab_size, seq_len, batch_size):

        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Inputs
        self.Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])
        self.temperature = tf.placeholder_with_default(1.0, shape=[])

        # Initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # linear output layer of the LSTM
        self.W_out = tf.get_variable("W_out", shape=(
            h_dim, self.vocab_size), initializer=self.weight_initializer)
        self.b_out = tf.get_variable("b_out", shape=(1, self.vocab_size),
                                     initializer=self.const_initializer)

        # Build computational graph
        # TODO: cast all inputs to self, or keep this way??
        self._lstm_generator(Z_dim, y_dim, X_dim, batch_size)

    def _lstm_generator(self, Z_dim, y_dim, X_dim, batch_size):
        """Generate text with a while loop over a LSTM cell"""
        # Start token, -1 for now
        # Will be a matrix of zeros
        # TODO: make shape flexible
        start_token = tf.one_hot(-tf.ones(shape=[1, batch_size]), self.vocab_size)

        # Initial state will be "sample" Z and "condition" y, concatenated
        initial_state = tf.concat(axis=1, values=[self.Z, self.y])
        state_tuple = tfr.LSTMStateTuple(initial_state, initial_state)

        # Just a simple LSTM for starters
        self.lstm = tfr.BasicLSTMCell(initial_state.get_shape()[1])

        # Dynamic array to store output at every timestep
        self.samples = tf.TensorArray(
            tf.int32, 1, dynamic_size=True, infer_shape=False, clear_after_read=False)

        i = tf.constant(0, dtype=int32)

        # Loop that supplies LSTM output prediction as next input, and stores in Tensor array
        _, _, _, self.samples = tf.while_loop(self._rollout_cond, self._lstm_rollout,
                                              (i, state_tuple, start_token, self.samples))

        # Stack tensor array with samples to get a regular tensor as output
        # This tensor will contain integer index outputs (so not one-hot)
        self.samples = tf.transpose(self.samples.stack(), name="predictions")

    #-- Helpers ------------------------------------------------------------------------------------

    def _lstm_rollout(self, i, state_tuple, X, predictions):
        """Compute a single timestep trough an LSTM cell; returns the input for the next timestep."""
        output, state = self.lstm(X, state_tuple)
        logits = tf.add(tf.matmul(output, self.W_out), self.b_out, name="logits")
        next_X, predictions = self._prediction(logits, i, predictions)
        i = tf.add(i, 1)
        return i, state, next_X, predictions

    def _rollout_cond(self, i, state_tuple, X, predictions):
        "While loop stopping condition"
        return tf.less(i, self.seq_len)

    def _probability(self, logits):
        """"Return the probability vector for a single timestep."""
        probability = tf.nn.softmax(logits / self.temperature, name="probability")
        return probability

    def _prediction(self, logits, idx, predictions):
        """Return a one-hot prediction that can serve as input to the next LSTM time step."""
        sampling_dist = tf.distributions.Categorical(probs=self._probability(logits))
        prediction = sampling_dist.sample(name="prediction")
        predictions = predictions.write(idx, prediction)
        return tf.one_hot(prediction, self.vocab_size), predictions
