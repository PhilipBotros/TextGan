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

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_flags
from settings import parse_arguments

g_sequence_len = 30
SAVE_PATH = 'generator_char.pt'
SAVE_EVERY = 100

opt = parse_arguments()
print_flags(opt)

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

if opt.das:
    POSITIVE_FILE = '/home/pbotros/TextGan/data/real_char.data'
    idx_to_char = create_vocab_dict("/home/pbotros/TextGan/data/vocabulary_char.txt")
else:
    POSITIVE_FILE = '../../data/real_char.data'
    idx_to_char = create_vocab_dict("../../data/vocabulary_char.txt")


with open('../../idx_to_char.json', 'r') as f:
    idx_to_char = json.load(f)

SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 100000
VOCAB_SIZE = 99
NR_EPOCHS = 100000

real_data = read_file(POSITIVE_FILE, g_sequence_len)

print(''.join([idx_to_char[str(idx)] for idx in real_data[10]]))

g_emb_dim = 32
g_hidden_dim = 32

# Define Networks
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

if os.path.isfile(SAVE_PATH):
    generator.load_state_dict(torch.load(SAVE_PATH))

if opt.cuda:
    generator = generator.cuda()

# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
gen_optimizer = optim.Adam(generator.parameters())
if opt.cuda:
    gen_criterion = gen_criterion.cuda()

gen_data_iter = GenDataIter(real_data, BATCH_SIZE)

print('Pretrain with MLE ...')
for i in range(NR_EPOCHS):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer,
                         BATCH_SIZE, opt.cuda)
    print('Epoch [%d] Model Loss: %f'% (i, loss))
    samples = generator.sample(BATCH_SIZE, g_sequence_len)

    # Print some samples
    for j in range(10):
        print(''.join([idx_to_char[str(idx)] for idx in samples.data[j]]))


    if i % SAVE_EVERY == 0:
        torch.save(generator.state_dict(), SAVE_PATH)
