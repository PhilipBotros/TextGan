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
from helpers import create_vocab_dict, print_samples


def sample_gen(opt, data_path):
    if opt.cuda is not None and opt.cuda >= 0:
        # Enable GPU
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True

    # Load vocab dict
    if opt.mode == 'word':
        idx_to_word = create_vocab_dict(os.path.join(data_path, 'idx_to_word.json'))
    elif opt.mode == 'char':
        idx_to_word = create_vocab_dict(os.path.join(data_path, 'idx_to_char.json'))
    else:
        raise Exception('Mode not recognized.')

    # Default model path
    if opt.gen_path is None:
        opt.gen_path = 'generator_attention.pt'

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
    else:
        raise Exception("No generator specified, please use flag --gen_path to specify which generator to load.")

    if opt.cuda:
        generator = generator.cuda()

    print('Sampling ...')
    samples = generator.sample(opt.batch_size, opt.seq_len)
    print_samples(opt.batch_size, idx_to_word, samples)


if __name__ == '__main__':
    # Path to data folder
    data_path = os.path.join(os.getcwd(), '../../data')

    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    # Pre-train the generator
    sample_gen(opt, data_path)
