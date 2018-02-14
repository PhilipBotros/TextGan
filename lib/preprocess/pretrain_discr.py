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


# Parse settings
opt = parse_arguments()
print_flags(opt)

# Enable GPU
if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Default data paths
if opt.positive_file is None:
    opt.positive_file = os.path.join(os.getcwd(), '../../data/real_char.data')

# Load vocab dict
if opt.mode == 'word':
    idx_to_word = create_vocab_dict(os.path.join(os.getcwd(), '../../data/idx_to_word.json'))
else if opt.mode == 'char':
    idx_to_word = create_vocab_dict(os.path.join(os.getcwd(), '../../data/idx_to_char.json'))
else:
    raise Exception('Mode not recognized.')

# Read data file
real_data = read_file(opt.positive_file, opt.seq_len)

# Default model paths
if opt.gen_path is None:
    opt.gen_path = 'generator_char.pt'
if opt.dis_path is None:
    opt.dis_path = 'discriminator_char.pt'


# One-hot encodings with character LSTM's
if opt.mode == 'char':
    opt.emb_dim = opt.vocab_size

generator = Generator(opt.vocab_size, opt.gen_hid_dim, opt.emb_dim, opt.num_layers, opt.cuda, opt.mode)
discriminator = Discriminator(opt.num_class, opt.vocab_size, opt.dis_hid_dim,
                              opt.emb_dim, opt.num_layers, opt.cuda, opt.mode)

if os.path.isfile(opt.gen_path):
    generator.load_state_dict(torch.load(opt.gen_path))
if os.path.isfile(opt.dis_path):
    discriminator.load_state_dict(torch.load(opt.dis_path))

# Pretrain Discriminator
dis_criterion = nn.NLLLoss(size_average=False)
parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
dis_optimizer = optim.Adam(parameters)

if opt.cuda:
    dis_criterion = dis_criterion.cuda()
    generator = generator.cuda()
    discriminator = discriminator.cuda()

print('Pretrain Discriminator...')

for i in range(opt.num_epochs):
    samples = generate_samples(generator, opt.batch_size, opt.num_gen, opt.seq_len)
    dis_data_iter = DisDataIter(real_data, samples, opt.batch_size, opt.lstm_rewards)
    for _ in range(1):
        loss = train_epoch(discriminator, dis_data_iter,
                           dis_criterion, dis_optimizer, opt.batch_size, opt.cuda, opt.lstm_rewards)
        print('Epoch [%d], loss: %f' % (i, loss))

    if i % opt.save_every == 0:
        torch.save(discriminator.state_dict(), opt.dis_path)
