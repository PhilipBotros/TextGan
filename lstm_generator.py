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
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.y = tf.one_hot(self.y, y_dim)

        self.i = 0
        self.temperature = tf.placeholder_with_default(1.0, shape=[])

        # Initial state will be "sample" Z and "condition" y, concatenated
        self.initial_state = tf.concat(axis=1, values=[self.Z, tf.cast(self.y, tf.float32)])
        self.h_dim = Z_dim + y_dim

        self.start_token = tf.one_hot(
        tf.placeholder(tf.int32, shape=[None], name="start_token"), self.vocab_size, dtype=tf.float32)

        # Initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # linear output layer of the LSTM
        self.W_out = tf.get_variable("G_W_out", shape=(
            self.h_dim, self.vocab_size), initializer=self.weight_initializer)
        self.b_out = tf.get_variable("G_b_out", shape=(1, self.vocab_size),
                                     initializer=self.const_initializer)

        # Build computational graph
        # TODO: cast all inputs to self, or keep this way??
        self.LSTM = tfr.BasicLSTMCell(self.h_dim)


        self.init_state = tfr.LSTMStateTuple(self.initial_state, self.initial_state)
        _, _, lstm_vars = self._lstm_generator(self.start_token, self.init_state, self.i)

        self.theta_G = [self.W_out, self.b_out, lstm_vars]

    def _lstm_generator(self, start_token, state, i):
        """Generate text with a while loop over a LSTM cell"""

        # Dynamic array to store output at every timestep
        self.samples = tf.TensorArray(
            tf.int32, 1, dynamic_size=True, infer_shape=False, clear_after_read=False)

        self.states = tf.TensorArray(
            tf.float32, self.h_dim, dynamic_size=True, infer_shape=False, clear_after_read=False)

        i = tf.constant(i, dtype=tf.int32)

        with tf.variable_scope("Generator") as scope:
            # Loop that supplies LSTM output prediction as next input, and stores in Tensor array
            _, self.states, _, self.samples = tf.while_loop(self._rollout_cond, self._lstm_rollout,
                                                  (i, self.init_state, start_token, self.samples))

            # Stack tensor array with samples to get a regular tensor as output
            # This tensor will contain integer index outputs (so not one-hot)
            self.samples = tf.transpose(self.samples.stack(), name="predictions")

            # We have a slight problem here
            # self.states = tf.transpose(self.states.stack(), name="hidden_states")

            trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        return self.samples, self.states, trainable_vars

    #-- Helpers ------------------------------------------------------------------------------------

    def _lstm_rollout(self, i, state_tuple, X, predictions):
        """Compute a single timestep trough an LSTM cell; returns the input for the next timestep."""
        # with tf.variable_scope("LSTM", reuse=True) as scope:
        output, new_state = self.LSTM(X, state_tuple)
        logits = tf.add(tf.matmul(output, self.W_out), self.b_out, name="logits")
        next_X, predictions = self._prediction(logits, i, predictions)
        i = tf.add(i, 1)
        return i, new_state, next_X, predictions

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

    def loss(self, samples, states, discriminator, nr_rollouts = 5):
        """Generator loss. Based on the expected reward given by MC rollout"""
        # Rollout for every t < T
        # Put through discriminator to get score

        loss = 0
        for t in range(self.seq_len - 1):
            for _ in range(nr_rollouts):
                # rollout, _, _ = self._lstm_generator(tf.cast(samples[t], tf.float32), states[t], t)
                rollout, _, _ = self._lstm_generator(tf.cast(samples[t], tf.float32), self.init_state, t)
                rollout_seq = tf.concat(samples[:t] + rollout, axis=1)
                loss += tf.reduce_sum(discriminator.inference(rollout_seq, self.y))
        
        # Get average loss over N rollouts
        loss /= nr_rollouts                
        # Add loss of complete sequence (t = T)
        loss += tf.reduce_sum(discriminator.inference(samples, self.y))

        return tf.log(tf.reduce_sum(loss))