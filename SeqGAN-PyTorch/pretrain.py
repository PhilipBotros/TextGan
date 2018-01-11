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

g_sequence_len = 8
SAVE_PATH = 'generator.pt'

# the_model.load_state_dict(torch.load(PATH))
# generator = model.load_state_dict(torch.load(SAVE_PATH))

def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')

        # Only load sequences of the set length
        if len(l) == g_sequence_len:
            try:
                # Catch faulty sentences
                l = [int(s) for s in l]
            except:
                continue
            lis.append(l)

    return lis

def create_vocab_dict(vocab_file):
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    dic = {}
    for i, line in enumerate(lines):
        dic[i] = line.strip()
    return dic

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

idx_to_word = create_vocab_dict("../data/vocabulary.txt")
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 1000
GENERATED_NUM = 10000
VOCAB_SIZE = 5003
PRE_EPOCH_NUM = 1
POSITIVE_FILE = '../data/real.data'
real_data = read_file(POSITIVE_FILE)

g_emb_dim = 32
g_hidden_dim = 32

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
        if data.shape[0] != BATCH_SIZE:
            continue
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

    return total_loss / total_words

# Define Networks
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

if os.path.isfile(SAVE_PATH):
    generator.load_state_dict(torch.load(SAVE_PATH))

if opt.cuda:
    generator = generator.cuda()
# Generate toy data using target lstm
nr_epochs = 100000
# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
gen_optimizer = optim.Adam(generator.parameters())
if opt.cuda:
    gen_criterion = gen_criterion.cuda()

gen_data_iter = GenDataIter(real_data, BATCH_SIZE)

print('Pretrain with MLE ...')
for epoch in range(nr_epochs):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
    print('Epoch [%d] Model Loss: %f'% (epoch, loss))
    samples = generator.sample(BATCH_SIZE, g_sequence_len)

    # Print some samples
    for i in range(10):
        print(' '.join([idx_to_word[idx] for idx in samples.data[i]]))

torch.save(generator.state_dict(), SAVE_PATH)