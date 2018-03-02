# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from feedforward import FeedForward, FeedForwardSum


class Generator(nn.Module):
    """Generator with self attention"""

    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_layers, batch_size, seq_len, use_cuda, mode='word', start_token=0, att_type='sum'):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.start_token = start_token
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.att_type = att_type

        # Define embeddings
        self.emb = nn.Embedding(vocab_size, embedding_dim)

        if mode == "char":
            if vocab_size == embedding_dim:
                # One hot encodings
                self.emb.weight.data = torch.eye(vocab_size)
                self.emb.weight.requires_grad = False

        # Set number of layers to 1
        self.num_layers = 1

        # Create encoder
        self.lstm_enc = nn.LSTMCell(embedding_dim, hidden_dim)

        # Create decoder
        # Context dim == hidden dim
        self.lstm_dec = nn.LSTMCell(hidden_dim, hidden_dim)
        self.linear_dec = nn.Linear(hidden_dim, vocab_size)

        # Init alignment model
        # We can precompute Ua * hj to save computation, CHECK PAPER
        # In our case we can store it until timestep I think
        # SUM OVER 2 input layers
        if self.att_type == 'cat':
            self.alignment_model = FeedForward(hidden_dim + hidden_dim, hidden_dim, 1)
        elif self.att_type == 'sum':
            self.alignment_model = FeedForwardSum(hidden_dim, hidden_dim, 1)
        else:
            raise(Exception('Attention model type not recognized, use --att_type to specify sum or cat.'))

        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.init_params()

    def forward(self, x):
        """
        A forward pass of a batch of sentences through the Encoder-Decoder AttnGan.
        Args:
            x: (batch_size, seq_len), input sequences
        """
        # Intialize hidden states and storage
        h_t_enc, c_t_enc = self.init_hidden(self.batch_size)
        h_t_dec, c_t_dec = self.init_hidden(self.batch_size)
        annotations = list()
        outputs_dec = list()

        # Initialize context vector
        context_t = Variable(torch.zeros((self.batch_size, self.hidden_dim)))
        if self.use_cuda:
            context_t = context_t.cuda()

        for i in range(self.seq_len):
            # One timestep through encoder-decoder LSTM
            output_dec, h_t_enc, c_t_enc, h_t_dec, c_t_dec, context_t = self.step(
                x[:, i], context_t, h_t_enc, c_t_enc, h_t_dec, c_t_dec, i, annotations)

            # Store annotations (hidden states encoder) and outputs per timestep
            annotations.append(h_t_enc)
            outputs_dec += [output_dec]

        # Reduce to ((batch_size x seq_len), vocab_size) for smooth log likelihood computation
        outputs = torch.stack(outputs_dec, 1).view(-1, self.vocab_size)

        return outputs

    def step(self, x, context_t, h_t_enc, c_t_enc, h_t_dec, c_t_dec, t, annotations, sample=False):
        """
        A single timestep of the Encoder-Decoder LSTM.
        Args:
            x: (batch_size,  1), batch of input words for current timestep
            context_t: Pre-initialized torch variable that will contain the context vector.
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        # (batch_size x embedding_dim)
        emb = self.emb(x)

        # Put embeddings of current timestep into the encoder
        h_t_enc, c_t_enc = self.lstm_enc(emb, (h_t_enc, c_t_enc))

        # Start updating the context vector after first timestep
        if t > 0:
            # List of unnormalized alignment vectors
            e_t = list()
            # Loop over all timesteps - 1 (preceding words)
            for j in range(t):
                if self.att_type == 'cat':
                    e_tj = self.alignment_model(torch.cat((h_t_dec, annotations[j]), 1))
                elif self.att_type == 'sum':
                    e_tj = self.alignment_model(h_t_dec, annotations[j])

                e_t.append(e_tj)

            # Create alignment vector for all elements in the batch
            # (Batch_size x timestep - 1)
            e_t = torch.stack(e_t, 1).squeeze(2)
            a_t = self.softmax(e_t)

            # Stack hidden states up until timestep t
            # (batch_size x hidden_size x timestep - 1)
            hidden_state = torch.stack(annotations, 2)
            # Unpack tensor (batch_size x hidden_size * (timestep - 1))
            hidden_state = hidden_state.contiguous().view(self.batch_size, -1)

            a_t = a_t.repeat(1, self.hidden_dim)

            # Compute context vector for every example in the batch
            context_t = a_t * hidden_state

            # Get back to 3 dimensional tensor and sum over the time dimension for context vector
            # (Batch_size x hidden_size)
            context_t = a_t.view(self.batch_size, self.hidden_dim, t)
            context_t = torch.sum(context_t, dim=2)

        # Give context vector (batch_size x hidden_dim) as input to the decoder
        h_t_dec, c_t_dec = self.lstm_dec(context_t, (h_t_dec, c_t_dec))

        # When we sample, we wish to return the respective probabilities, for the loss we need the log-probs
        if sample:
            pred = self.softmax(self.linear_dec(h_t_dec))
        else:
            pred = self.log_softmax(self.linear_dec(h_t_dec))

        return pred, h_t_enc, c_t_enc, h_t_dec, c_t_dec, context_t

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_params(self):
        for name, param in self.named_parameters():
            print(name, type(param.data), param.size())
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, seq_len, x=None):
        res = []
        flag = False  # whether to sample from zero-state
        if x is None:
            flag = True
        if flag:
            x = Variable(self.start_token * torch.ones((batch_size, 1)).long())

        # Intialize hidden states and storage
        h_t_enc, c_t_enc = self.init_hidden(self.batch_size)
        h_t_dec, c_t_dec = self.init_hidden(self.batch_size)
        context_t = Variable(torch.zeros((self.batch_size, self.hidden_dim)))
        annotations = list()
        outputs_dec = list()
        samples = []

        if self.use_cuda:
            x = x.cuda()
            context_t = context_t.cuda()

        if flag:
            for i in range(seq_len):
                output, h_t_enc, c_t_enc, h_t_dec, c_t_dec, context_t = self.step(
                    x, context_t, h_t_enc, c_t_enc, h_t_dec, c_t_dec, i, annotations, sample=True)
                x = output.multinomial(1)
                annotations.append(h_t_enc)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h_t_enc, c_t_enc, h_t_dec, c_t_dec, context_t = self.step(
                    lis[i], context_t, h_t_enc, c_t_enc, h_t_dec, c_t_dec, i, annotations, sample=True)
                annotations.append(h_t_enc)
            samples.extend(lis)
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h_t_enc, c_t_enc, h_t_dec, c_t_dec, context_t = self.step(
                    x, context_t, h_t_enc, c_t_enc, h_t_dec, c_t_dec, i, annotations, sample=True)
                annotations.append(h_t_enc)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)

        return output
