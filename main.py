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
import sys
sys.path.append("lib/model")
sys.path.append("lib/scripts")
sys.path.append("lib/utilities")

from train_GAN import train
from pretrain_gen import pretrain_gen
from pretrain_dis import pretrain_dis
from helpers import print_flags
from settings import parse_arguments
from create_vocab import create_vocab


if __name__ == '__main__':

    # Path to data folder
    data_path = os.path.join(os.getcwd(), 'data/')

    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    if not os.path.isfile(data_path + 'real_content.data'):
        create_vocab(opt)

    if bool(opt.pre_gen):
        pretrain_gen(opt, data_path)

    if bool(opt.pre_dis):
        pretrain_dis(opt, data_path)

    # Train the GAN
    if bool(opt.train_gan):
        train(opt, data_path)
