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
    parser.add_argument('--remote', default=0, type=int, help="True when we run on a server.")

    # Path settings
    parser.add_argument('--positive_file', default=None, type=str,
                        help="Path to file that contains real sequences.")
    parser.add_argument('--vocab_file', default=None, type=str,
                        help="Path to file that contains training vocabulary.")
    parser.add_argument("--gen_path", default=None, type=str,
                        help="Path to file that contains pretrained Generator")
    parser.add_argument("--sample_file", default="samples.txt", type=str,
                        help="Path to file that stores samples")
    parser.add_argument("--loss_file", default="loss.txt", type=str,
                        help="Path to file that stores loss")
    parser.add_argument("--dis_path", default=None, type=str,
                        help="Path to file that contains pretrained Discriminator")

    # Training settings
    parser.add_argument('--seed', default=88, type=int, help="Random seed.")
    parser.add_argument('--lstm_rewards', default=0, type=int,
                        help="Whether the rewards stem directly from LSTM per-word output. When False Monte Carlo search is used instead.")
    parser.add_argument('--batch_size', default=128, type=int, help="Number of sequences per batch.")
    parser.add_argument('--num_epochs', default=100, type=int,
                        help="Number of Generator/Discriminator epochs.")
    parser.add_argument('--num_gen', default=100, type=int,
                        help="How much fake sequences to generate per Discriminator epoch.")
    parser.add_argument('--vocab_size', default=30003, type=int,
                        help="Number of characters/words.")
    parser.add_argument('--num_class', default=2, type=int,
                        help="Number of Discriminator output classes.")
    parser.add_argument('--seq_len', default=5, type=int, help="Sequence length.")
    parser.add_argument('--save_every', default=1, type=int,
                        help="Save every X number of epochs.")

    # Model settings
    parser.add_argument('--num_layers', default=2, type=int,
                        help="Number of layers of the models.")
    parser.add_argument('--emb_dim', default=512, type=int, help="Word Embedding size.")
    parser.add_argument('--gen_hid_dim', default=256, type=int,
                        help="Hidden layer size of the Generator.")
    parser.add_argument('--dis_hid_dim', default=128, type=int,
                        help="Hidden layer size of the Discriminator.")
    parser.add_argument('--mode', default='word', type=str, help="Switch between 'char' and 'word' LSTM's.")
    parser.add_argument('--attention', default=1, type=int, help="Switch between LSTM's with or without attention.")
    parser.add_argument('--att_type', default='sum', type=str, help="Type of attention model, either 'sum' or 'cat'.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate of the optimizer")

    # Main script settings
    parser.add_argument('--pre_gen', default=1, type=int, help="1: Pretrain the generator.")
    parser.add_argument('--pre_dis', default=0, type=int, help="1: Pretrain the discriminator.")
    parser.add_argument('--train_gan', default=1, type=int, help="1: Adverserial training.")

    return parser.parse_args()
