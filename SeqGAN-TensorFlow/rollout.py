def get_reward(self, sess, input_x, rollout_num, discriminator):
    rewards = []
    for i in range(rollout_num):
        for given_num in range(1, 20):
            feed = {self.x: input_x, self.given_num: given_num}
            samples = sess.run(self.gen_x, feed)
            feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred

        # the last token reward
        feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[19] += ypred

    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
    return rewards