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


# Parse model settings
opt = parse_arguments()
print_flags(opt)

if opt.cuda is not None and opt.cuda >= 0:
    # Enable GPU
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Default data paths
if opt.positive_file is None:
    if opt.remote:
        opt.positive_file = '$HOME/TextGan/data/real.data'
        idx_to_word = create_vocab_dict("$HOME/TextGan/data/vocabulary.txt")
    else:
        opt.positive_file = '../../data/real.data'
        idx_to_word = create_vocab_dict("../../data/vocabulary.txt")

# Load real data
real_data = read_file(opt.positive_file, opt.seq_len)

# Define Networks
generator = Generator(opt.vocab_size, opt.gen_emb_dim, opt.gen_hid_dim, opt.cuda)

if os.path.isfile(opt.gen_path):
    generator.load_state_dict(torch.load(opt.gen_path))

if opt.cuda:
    generator = generator.cuda()

# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
gen_optimizer = optim.Adam(generator.parameters())
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
    for j in range(10):
        print(' '.join([idx_to_word[idx] for idx in samples.data[j]]))

    if i % opt.save_every == 0:
        torch.save(generator.state_dict(), opt.gen_path)
