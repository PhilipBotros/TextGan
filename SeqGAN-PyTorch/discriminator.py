# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Discriminator(nn.Module):
    """An LSTM for text classification

    architecture: Embedding >> LSTM >> Linear >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, hidden_dim, use_cuda):
        super(Discriminator, self).__init__()

        # Settings
        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim
        self.num_classes = num_classes
        self.use_cuda = use_cuda

        # Layers
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(emb_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Model initialization
        self.init_params()

    def forward(self, x, full):
        """
        Args:
            x: (batch_size, seq_len)
            full: False when only the last output should be returned
        Out:
            pred: (batch_size * seq_len, num_classes) or (batch_size, num_classes)
        """
        embeddings = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        self.init_hidden(x.data.shape[0])
        lstm_out, _ = self.lstm(embeddings, self.hidden)

        if full:
            logits = self.linear(lstm_out.contiguous().view(-1, self.hidden_dim))
            pred = self.softmax(logits)
        else:
            logits = self.linear(lstm_out[:, -1, :])
            pred = self.softmax(logits)

        return pred

    def init_hidden(self, batch_size):
        """
        Hidden state zero initialization for a single layer LSTM.
        """
        if self.use_cuda:
            self.hidden = (Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(),
                           Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda())
        else:
            self.hidden = (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                           Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
