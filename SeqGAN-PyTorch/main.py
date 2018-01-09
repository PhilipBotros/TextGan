
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
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
import utils
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 100
GENERATED_NUM = 10000
POSITIVE_FILE = 'real.data'
VOCAB_SIZE = 16
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


def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis


def generate_samples(model, batch_size, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    return samples


def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()

    return math.exp(total_loss / total_words)


def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
    data_iter.reset()

    return math.exp(total_loss / total_words)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adverserial training of Generator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)

        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, BATCH_SIZE,
                                  d_hidden_dim)

    # real_data = read_file(POSITIVE_FILE)

    real_data = utils.generate_fibonacci_batch(9984, g_sequence_len)

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
            print(samples[:10])
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 5, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            if opt.cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            prob = generator.forward(inputs)

            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        rollout.update_params()

        for _ in range(1):
            samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM)
            dis_data_iter = DisDataIter(list(real_data), samples, BATCH_SIZE)
            for _ in range(1):
                discriminator.init_hidden(BATCH_SIZE)
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
                print('Batch [%d] Loss: %f' % (total_batch, loss))


if __name__ == '__main__':
    main()
