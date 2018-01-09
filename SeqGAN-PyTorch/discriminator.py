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

    def __init__(self, num_classes, vocab_size, emb_dim, batch_size, hidden_dim):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(emb_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_hidden(batch_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        embeddings = self.embedding(x).view(x.data.shape[1], x.data.shape[0],
                                            self.embedding_dim)  # seq_len * batch_size * emb_dim
        lstm_out, hidden = self.lstm(embeddings, self.hidden)
        logits = self.linear(lstm_out[-1, :, :])
        pred = self.softmax(logits)
        return pred

    def init_hidden(self, batch_size):
        """
        Hidden state zero initialization for a single layer LSTM.
        """
        self.hidden = (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                       Variable(torch.zeros(1, batch_size, self.hidden_dim)))
