# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    """Generator """

    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_layers, batch_size, seq_len, use_cuda, mode='word', start_token=0):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.start_token = start_token

        # Defnine embeddings
        self.emb = nn.Embedding(vocab_size, embedding_dim)

        if mode == "char":
            if vocab_size == embedding_dim:
                # One hot encodings
                self.emb.weight.data = torch.eye(vocab_size)
                self.emb.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.init_params()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(emb, (h0, c0))
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        print("Forward:")
        print(pred.size())
        return pred

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=-1)
        return pred, h, c

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, seq_len, x=None):
        res = []
        flag = False  # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = Variable(self.start_token * torch.ones((batch_size, 1)).long())
        if self.use_cuda:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output
