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

SAVE_EVERY = 100

opt = parse_arguments()
print_flags(opt)

BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 10000
VOCAB_SIZE = 5003
NR_EPOCHS = 100000

g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 5

# Discriminator Parameters
d_emb_dim = 64
d_hidden_dim = 32
d_num_class = 2

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

if opt.das:
    POSITIVE_FILE = '/home/pbotros/TextGan/data/real.data'
    idx_to_word = create_vocab_dict("/home/pbotros/TextGan/data/vocabulary.txt")
else:
    POSITIVE_FILE = '../../data/real.data'
    idx_to_word = create_vocab_dict("../../data/vocabulary.txt")

GEN_PATH = 'generator.pt'
SAVE_PATH = 'discriminator.pt'
real_data = read_file(POSITIVE_FILE, g_sequence_len)

generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim,
                              d_hidden_dim, opt.cuda)
if os.path.isfile(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH))

# Pretrain Discriminator
dis_criterion = nn.NLLLoss(size_average=False)
dis_optimizer = optim.Adam(discriminator.parameters())

if opt.cuda:
    dis_criterion = dis_criterion.cuda()

print('Pretrain Discriminator...')

for i in range(NR_EPOCHS):
    samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, g_sequence_len)
    dis_data_iter = DisDataIter(real_data, samples, BATCH_SIZE, opt.lstm_rewards)
    for _ in range(1):
        loss = train_epoch(discriminator, dis_data_iter,
                           dis_criterion, dis_optimizer, BATCH_SIZE, opt.cuda, opt.lstm_rewards)
        print('Epoch [%d], loss: %f' % (i, loss))

    if i % SAVE_EVERY == 0:
        torch.save(generator.state_dict(), SAVE_PATH)
