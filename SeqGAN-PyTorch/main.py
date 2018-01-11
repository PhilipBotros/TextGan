
import os
import random
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch
from loss import GANLoss

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--full', action='store', default=False, type=bool)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 10000
POSITIVE_FILE = '../data/real.data'
VOCAB_SIZE = 5003
PRE_EPOCH_NUM = 1

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Generator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 5

# Discriminator Parameters
d_emb_dim = 64
# d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# d_filter_sizes = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]
# d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_hidden_dim = 32

# d_dropout = 0.75
d_num_class = 2

#==========================================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    idx_to_word = create_vocab_dict("../data/vocabulary.txt")

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim,
                                  d_hidden_dim, opt.cuda)

    real_data = read_file(POSITIVE_FILE, g_sequence_len)

    # real_data = utils.generate_fibonacci_batch(9984, g_sequence_len)

    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Adversarial Training
    rollout = Rollout(generator, 0.8)
    print('#####################################################')
    print('Start Adverserial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(size_average=False)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, g_sequence_len)

            # Print some samples
            for i in range(10):
                print(' '.join([idx_to_word[idx] for idx in samples.data[i]]))

            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 5, discriminator, opt.full)
            rewards = Variable(torch.Tensor(rewards)).contiguous().view((-1,))
            if opt.cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            prob = generator.forward(inputs)

            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        rollout.update_params()

        for _ in range(1):
            samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, g_sequence_len)
            dis_data_iter = DisDataIter(real_data, samples, BATCH_SIZE, opt.full)
            for _ in range(1):
                loss = train_epoch(discriminator, dis_data_iter,
                                   dis_criterion, dis_optimizer, BATCH_SIZE, opt.cuda, opt.full)
                print('Batch [%d] Loss: %f' % (total_batch, loss))


if __name__ == '__main__':
    main()
