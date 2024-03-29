# -*- coding:utf-8 -*-

import os
import random
import math
import copy
import time

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Rollout(object):
    """Roll-out policy"""

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, discriminator, full):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
            full: False -> MC Rollout, True -> LSTM rewards
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)

        if full:
            discriminator.init_hidden(batch_size)

            # Reward = probability of sample being true data in format [batch_size, seq_len]
            pred = discriminator(x, full).cpu().data[:, 1].contiguous().view(-1, seq_len).numpy()

            for l in range(0, seq_len):
                rewards.append(pred[:, l])
        else:
            for i in range(num):
                for l in range(1, seq_len):
                    data = x[:, 0:l]
                    samples = self.own_model.sample(batch_size, seq_len, data)
                    discriminator.init_hidden(batch_size)
                    pred = discriminator(samples.squeeze(2), full)
                    pred = pred.cpu().data[:, 1].numpy()
                    if i == 0:
                        rewards.append(pred)
                    else:
                        rewards[l - 1] += pred

                # for the last token
                pred = discriminator(x, full)
                pred = pred.cpu().data[:, 1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[seq_len - 1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)  # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
