import os
import random
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import sys
sys.path.append('../model')
sys.path.append('../utilities')
sys.path.append('../../')

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_flags
from settings import parse_arguments

SAVE_EVERY = 1

opt = parse_arguments()
print_flags(opt)

BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 10000
VOCAB_SIZE = 100
NR_EPOCHS = 100000

g_emb_dim = 32
g_hidden_dim = 128
g_sequence_len = 30
g_num_layers = 2

# Discriminator Parameters
d_emb_dim = 64
d_hidden_dim = 128
d_num_class = 2

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

POSITIVE_FILE = os.path.join(os.getcwd(), '../../data/real_char.data')
idx_to_char = create_vocab_dict(os.path.join(os.getcwd(), '../../data/idx_to_char.json'))

GEN_PATH = 'generator_char.pt'
SAVE_PATH = 'discriminator_char.pt'
real_data = read_file(POSITIVE_FILE, g_sequence_len)

generator = Generator(VOCAB_SIZE, g_hidden_dim, g_num_layers, opt.cuda)
discriminator = Discriminator(d_num_class, VOCAB_SIZE,
                              d_hidden_dim, opt.cuda, g_num_layers)
if os.path.isfile(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH))

if os.path.isfile(SAVE_PATH):
    discriminator.load_state_dict(torch.load(SAVE_PATH))

# Pretrain Discriminator
dis_criterion = nn.NLLLoss(size_average=False)
parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
dis_optimizer = optim.Adam(parameters)

if opt.cuda:
    dis_criterion = dis_criterion.cuda()
    generator = generator.cuda()
    discriminator = discriminator.cuda()

print('Pretrain Discriminator...')

for i in range(NR_EPOCHS):
    samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, g_sequence_len)
    dis_data_iter = DisDataIter(real_data, samples, BATCH_SIZE, opt.lstm_rewards)
    for _ in range(1):
        loss = train_epoch(discriminator, dis_data_iter,
                           dis_criterion, dis_optimizer, BATCH_SIZE, opt.cuda, opt.lstm_rewards)
        print('Epoch [%d], loss: %f' % (i, loss))

    if i % SAVE_EVERY == 0:
        torch.save(discriminator.state_dict(), SAVE_PATH)
