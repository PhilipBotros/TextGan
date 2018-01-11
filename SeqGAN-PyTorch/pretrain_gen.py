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

g_sequence_len = 5
SAVE_PATH = 'generator.pt'
SAVE_EVERY = 100

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--full', action='store', default=True, type=bool)
parser.add_argument('--das', action='store', default=False, type=bool)
opt = parser.parse_args()
print(opt)

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

if opt.das:
    POSITIVE_FILE = '/home/pbotros/TextGan/data/real.data'
    idx_to_word = create_vocab_dict("/home/pbotros/TextGan/data/vocabulary.txt")
else:
    POSITIVE_FILE = '../data/real.data'
    idx_to_word = create_vocab_dict("../data/vocabulary.txt")

SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 10000
VOCAB_SIZE = 5003
PRE_EPOCH_NUM = 1

real_data = read_file(POSITIVE_FILE, g_sequence_len)

g_emb_dim = 32
g_hidden_dim = 32

# Define Networks
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

if os.path.isfile(SAVE_PATH):
    generator.load_state_dict(torch.load(SAVE_PATH))

if opt.cuda:
    generator = generator.cuda()
# Generate toy data using target lstm
nr_epochs = 100
# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
gen_optimizer = optim.Adam(generator.parameters())
if opt.cuda:
    gen_criterion = gen_criterion.cuda()

gen_data_iter = GenDataIter(real_data, BATCH_SIZE)

print('Pretrain with MLE ...')
for i in range(nr_epochs):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer,
                         BATCH_SIZE, opt.cuda)
    print('Epoch [%d] Model Loss: %f'% (epoch, loss))
    samples = generator.sample(BATCH_SIZE, g_sequence_len)

    # Print some samples
    for j in range(10):
        print(' '.join([idx_to_word[idx] for idx in samples.data[j]]))

    if i % SAVE_EVERY == 0:
        torch.save(generator.state_dict(), SAVE_PATH)