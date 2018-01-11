import os
import random
import math
import argparse
import numpy as np

from torch.autograd import Variable


def read_file(data_file, seq_len):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')

        # Only load sequences of the set length
        if len(l) == seq_len:
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


def generate_samples(model, batch_size, generated_num, seq_len):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    return samples


def train_epoch(model, data_iter, criterion, optimizer, full, batch_size, is_cuda):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
        if data.shape[0] != batch_size:
            continue
        data = Variable(data)
        target = Variable(target)
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data, full)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()

    return total_loss / total_words


def eval_epoch(model, data_iter, criterion, is_cuda):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
    data_iter.reset()

    return math.exp(total_loss / total_words)
