import os
import random
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json

import sys
sys.path.append('../model')
sys.path.append('../utilities')
sys.path.append('../../')

from generator_att import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_flags
from settings import parse_arguments


# Parse model settings
opt = parse_arguments()
print_flags(opt)

if opt.cuda is not None and opt.cuda >= 0:
    # Enable GPU
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Default data paths
if opt.positive_file is None:
    opt.positive_file = os.path.join(os.getcwd(), '../../data/real.data')

# Load vocab dict
if opt.mode == 'word':
    idx_to_word = create_vocab_dict(os.path.join(os.getcwd(), '../../data/idx_to_word.json'))
elif opt.mode == 'char':
    idx_to_word = create_vocab_dict(os.path.join(os.getcwd(), '../../data/idx_to_char.json'))
else:
    raise Exception('Mode not recognized.')

# Load real data
real_data = read_file(opt.positive_file, opt.seq_len)

# Default model paths
if opt.gen_path is None:
    opt.gen_path = 'generator_char.pt'

# One hot encodings for character LSTMs
if opt.mode == "char":
    opt.emb_dim = opt.vocab_size

# Define Networks
generator = Generator(opt.vocab_size, opt.gen_hid_dim, opt.emb_dim, opt.num_layers, opt.batch_size, opt.seq_len, opt.cuda, opt.mode)

if os.path.isfile(opt.gen_path):
    generator.load_state_dict(torch.load(opt.gen_path))

if opt.cuda:
    generator = generator.cuda()

# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
parameters = filter(lambda p: p.requires_grad, generator.parameters())
gen_optimizer = optim.Adam(parameters)
if opt.cuda:
    gen_criterion = gen_criterion.cuda()

gen_data_iter = GenDataIter(real_data, opt.batch_size)

print('Pretrain with MLE ...')
for i in range(opt.num_epochs):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer,
                       opt.batch_size, opt.cuda)
    print('Epoch [%d] Model Loss: %f' % (i, loss))
    samples = generator.sample(opt.batch_size, opt.seq_len)

    # Print some samples
    print_samples(10, idx_to_word, samples, opt.mode)

    if i % opt.save_every == 0:
        torch.save(generator.state_dict(), opt.gen_path)
