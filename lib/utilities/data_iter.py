# -*- coding:utf-8 -*-

import os
import random
import time
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GenDataSet(Dataset):
    """"DataSet to load sentences"""

    def __init__(self, data_lis):
        self.data = torch.from_numpy(np.asarray(data_lis, dtype=np.int64))
        self.data_len = len(data_lis)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        datapoint = self.data[idx, 0:-1]
        target = self.data[idx, 1:]
        return datapoint, target


class GenDataIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, data_lis, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = data_lis
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        if d.shape[0] != self.batch_size:
            raise StopIteration
        # data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        # target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        data = d[:, 0:-1]
        target = d[:, 1:]
        self.idx += self.batch_size
        return data, target


class DisDataIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, real_data_lis, fake_data_lis, batch_size, full):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        self.data = list(np.asarray(real_data_lis)[:, 1:]) + fake_data_lis

        if full:
            # Target for every word in the sequence
            self.labels = [[1 for _ in range(len(real_data_lis[0]) - 1)] for _ in range(len(real_data_lis))] +\
                [[0 for _ in range(len(fake_data_lis[0]))] for _ in range(len(fake_data_lis))]
        else:
            # Only a target for every sequence
            self.labels = [1 for _ in range(len(real_data_lis))] +\
                [0 for _ in range(len(fake_data_lis))]

        self.pairs = list(zip(self.data, self.labels))
        random.shuffle(self.pairs)
        self.data_num = len(self.data)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size

        return data, label
