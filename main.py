"""
Training Script for SeqGAN on news titles.

The model is trained with the signalmedia news dataset.
With create_vocab.py a vocabulary of specified size and data file in the correct format
can be created from the raw signalmedia JSON. This model can be trained on many sequential (text) data;
to do so a new dataset handler (see vocabulary.py) has to be written that converts raw data to the
correct format (one-hot encoded sequences).

To pretrain the Discriminator and Generator pretrain_discr.py and pretrain.gen.py can be used
respectively. It is advisable to first pretrain a Generator, that provides fake data for pretraining
the Discriminator.

During training, the script will print ten generated samples every epoch as a measure
of sample quality. No quantitative measures have been implemented. The Discriminator loss is also
printed to the terminal, which is informative for the Discriminator-Generator balance (the loss
should neither be too high or too low).

Various model and training settings can be controlled via the command line. These can be found with
description in settings.py. A help message can also be displayed via the command line.

The model is built with the PyTorch neural network library and optimized for cuda.

Based on the Pytorch-SeqGAN implementation by ZiJianZhao (https://github.com/ZiJianZhao/SeqGAN-PyTorch).
Authors: Philip Botros and Tom Pelsmaeker
"""

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lib.model.generator import Generator
from lib.model.discriminator import Discriminator
from lib.model.rollout import Rollout
from lib.utilities.data_iter import GenDataIter, DisDataIter
from lib.utilities.helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_flags, print_samples
from lib.model.loss import GANLoss
from settings import parse_arguments


def main():
    """
    Training Script for SeqGAN on News titles.
    """

    if opt.cuda is not None and opt.cuda >= 0:
        # Enable GPU
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True

    if opt.positive_file is None:
        # Use default data paths if none are specified
        opt.positive_file = os.path.join(os.getcwd(), 'data/real.data')
        idx_to_word = os.path.join(os.getcwd(), 'data/idx_to_word.json')

    if opt.mode == 'char':
        # One hot encodings for character LSTMs
        opt.emb_dim = opt.vocab_size

    # Seed for the random number generators
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Data and vocabulary
    idx_to_word = create_vocab_dict(idx_to_word)
    real_data = read_file(opt.positive_file, opt.seq_len)

    # Define Networks
    generator = Generator(opt.vocab_size, opt.gen_hid_dim, opt.emb_dim, opt.num_layers, opt.cuda, opt.mode)
    discriminator = Discriminator(opt.num_class, opt.vocab_size, opt.dis_hid_dim,
                                  opt.emb_dim, opt.num_layers, opt.cuda, opt.mode)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Default model paths
    if opt.gen_path is None:
        opt.gen_path = 'generator_char.pt'
    if opt.dis_path is None:
        opt.dis_path = 'discriminator_char.pt'

    # Load pretrained Generator and Discriminator when provided
    if os.path.isfile(opt.gen_path):
        generator.load_state_dict(torch.load(opt.gen_path))
        print("Loading generator...")
    else:
        print("No pretrained Generator found; model starting from scratch.")
    if os.path.isfile(opt.dis_path):
        discriminator.load_state_dict(torch.load(opt.dis_path))
        print("Loading discriminator...")
    else:
        print("No pretrained Discriminator found; model starting from scratch.")

    # Loss and optimizer for the Generator and Discriminator
    gen_loss = GANLoss()
    gen_parameters = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = optim.Adam(gen_parameters)
    dis_loss = nn.NLLLoss(size_average=False)
    dis_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
    dis_optimizer = optim.Adam(dis_parameters)

    # Rollout policy; Monte Carlo search or rewards from LSTM
    rollout = Rollout(generator, 0.8)

    if opt.cuda:
        gen_loss = gen_loss.cuda()
        dis_loss = dis_loss.cuda()

    print('#####################################################')
    print('Start Adverserial Training...\n')
    for num_epochs in range(opt.num_epochs):
        # Train the Generator and Discriminator
        for it in range(1):
            # Generate some samples for printing
            samples = generator.sample(opt.batch_size, opt.seq_len)
            print_samples(10, idx_to_word, samples, opt.mode)

            # Construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((opt.batch_size, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))

            # Calculate the reward
            rewards = rollout.get_reward(samples, 5, discriminator, opt.lstm_rewards)
            rewards = Variable(torch.Tensor(rewards)).contiguous().view((-1,))
            if opt.cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))

            # Calculate the loss and backpropagate
            prob = generator.forward(inputs)
            loss = gen_loss(prob, targets, rewards)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()

        rollout.update_params()

        for _ in range(1):
            # Train the Discriminator for one step
            samples = generate_samples(generator, opt.batch_size, opt.num_gen, opt.seq_len)
            dis_data_iter = DisDataIter(real_data, samples, opt.batch_size, opt.lstm_rewards)
            loss = train_epoch(discriminator, dis_data_iter,
                               dis_loss, dis_optimizer, opt.batch_size, opt.cuda, opt.lstm_rewards)
            print('Batch [%d] Loss: %f' % (num_epochs, loss))

        if num_epochs % opt.save_every == 0:
            torch.save(discriminator.state_dict(), opt.dis_path)
            torch.save(generator.state_dict(), opt.gen_path)


if __name__ == '__main__':
    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    # Train the model
    main()
