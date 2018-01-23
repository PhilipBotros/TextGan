"""
This file contains the full settings of the SeqGAN model.
Changing the settings means changing the default behavior of the model.
Alternatively, the settings can be changed via command line when running main.py.

Copyright Tom Pelsmaeker and Philip Botros @ 2018
"""

import argparse


def parse_arguments():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training Parameters")

    # Device settings
    parser.add_argument('--cuda', default=None, type=int,
                        help="Device number of GPU, None when CPU is used.")
    parser.add_argument('--remote', default=False, type=bool, help="True when we run on a server.")

    # Path settings
    parser.add_argument('--positive_file', default=None, type=str,
                        help="Path to file that contains real sequences.")
    parser.add_argument('--vocab_file', default=None, type=str,
                        help="Path to file that contains training vocabulary.")
    parser.add_argument("--gen_path", default=None, type=str,
                        help="Path to file that contains pretrained Generator")
    parser.add_argument("--dis_path", default=None, type=str,
                        help="Path to file that contains pretrained Discriminator")

    # Training settings
    parser.add_argument('--seed', default=88, type=int, help="Random seed.")
    parser.add_argument('--lstm_rewards', default=False, type=bool,
                        help="Whether the rewards stem directly from LSTM per-word output. When False Monte Carlo search is used instead.")
    parser.add_argument('--batch_size', default=64, type=int, help="Number of sequences per batch.")
    parser.add_argument('--num_epochs', default=100, type=int,
                        help="Number of Generator/Discriminator epochs.")
    parser.add_argument('--num_gen', default=10000, type=int,
                        help="How much fake sequences to generate per Discriminator epoch.")
    parser.add_argument('--vocab_size', default=5003, type=int,
                        help="Number of words in vocabulary.")
    parser.add_argument('--num_class', default=2, type=int,
                        help="Number of Discriminator output classes.")
    parser.add_argument('--seq_len', default=5, type=int, help="Sequence length.")
    parser.add_argument('--save_every', default=100, type=int,
                        help="Save every X number of epochs.")

    # Model settings
    parser.add_argument('--cond', default=False, type=bool,
                        help="Whether to train a conditional TextGan")
    parser.add_argument('--gen_emb_dim', default=32, type=int,
                        help="Embedding size of the Generator.")
    parser.add_argument('--gen_hid_dim', default=32, type=int,
                        help="Hidden layer size of the Generator.")
    parser.add_argument('--dis_emb_dim', default=64, type=int,
                        help="Embedding size of the Discriminator")
    parser.add_argument('-dis_hid_dim', default=32, type=int,
                        help="Hidden layer size of the Discriminator")

    return parser.parse_args()
