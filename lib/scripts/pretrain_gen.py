# General python libraries
import os
import random

# Math libraries
import numpy as np

# Torch neural network libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Main script libraries
if __name__ == '__main__':
    import sys
    sys.path.append('../model')
    sys.path.append('../utilities')
    sys.path.append('../../')
    from settings import parse_arguments
    from helpers import print_flags

# Custom libraries
from generator_att import Generator as GeneratorAttention
from generator_timestep import Generator as GeneratorTimestep
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter, GenDataSet
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_samples, save_samples


def pretrain_gen(opt, data_path):
    if opt.cuda is not None and opt.cuda >= 0:
        # Enable GPU
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True

    # Default data paths
    if opt.positive_file is None:
        if opt.mode == 'word':
            opt.positive_file = os.path.join(data_path, 'real_content.data')
        elif opt.mode == 'char':
            opt.positive_file = os.path.join(data_path, 'real_char.data')

    # Load vocab dict
    if opt.mode == 'word':
        idx_to_word = create_vocab_dict(os.path.join(data_path, 'idx_to_word.json'))
    elif opt.mode == 'char':
        idx_to_word = create_vocab_dict(os.path.join(data_path, 'idx_to_char.json'))
    else:
        raise Exception('Mode not recognized.')

    # Load real data
    real_data = read_file(opt.positive_file, opt.seq_len)

    # Default model paths
    if opt.gen_path is None:
        opt.gen_path = 'generator_content.pt'

    # One hot encodings for character LSTMs
    if opt.mode == "char":
        opt.emb_dim = opt.vocab_size

    # Define Generator
    if opt.attention:
        print("Using attention")
        generator = GeneratorAttention(opt.vocab_size, opt.gen_hid_dim, opt.emb_dim, opt.num_layers,
                                       opt.batch_size, opt.seq_len, opt.cuda, opt.mode, att_type=opt.att_type)

    else:
        generator = Generator(opt.vocab_size, opt.gen_hid_dim, opt.emb_dim, opt.num_layers,
                              opt.batch_size, opt.seq_len, opt.cuda, opt.mode)

    if os.path.isfile(opt.gen_path):
        print("Loading generator")
        generator.load_state_dict(torch.load(opt.gen_path))

    if opt.cuda:
        generator = generator.cuda()

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(size_average=False)
    parameters = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = optim.Adam(parameters, opt.lr)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()

    # gen_data_iter = GenDataIter(real_data, opt.batch_size)
    gen_data_iter = DataLoader(GenDataSet(real_data), batch_size=opt.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)

    print('Pretrain with MLE ...')
    for i in range(opt.num_epochs):
        samples = generator.sample(opt.batch_size, opt.seq_len)

        # Print and save some samples
        print_samples(10, idx_to_word, samples)
        save_samples(10, idx_to_word, samples, opt.sample_file, i)

        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer,
                           opt.batch_size, opt.cuda)

        # Print and save loss
        print('Epoch [%d] Model Loss: %f' % (i, loss))
        with open(opt.loss_file, 'a') as f:
            f.write('Epoch [%d] Model Loss: %f\n' % (i, loss))

        if i % opt.save_every == 0:
            torch.save(generator.state_dict(), opt.gen_path)


if __name__ == '__main__':
    # Path to data folder
    data_path = os.path.join(os.getcwd(), '../../data')

    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    # Pre-train the generator
    pretrain_gen(opt, data_path)
