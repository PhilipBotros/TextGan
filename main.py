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
        if opt.das:
            opt.positive_file = '/var/scratch/pbotros/data/real.data'
            opt.vocab_file = '/var/scratch/pbotros/data/vocabulary.txt'
        else:
            opt.positive_file = './data/real.data'
            opt.vocab_file = './data/vocabulary.txt'

    # Seed for the random number generators
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Data and vocabulary
    idx_to_word = create_vocab_dict(opt.vocab_file)
    real_data = read_file(opt.positive_file, opt.seq_len)

    # Define Networks
    generator = Generator(opt.vocab_size, opt.gen_emb_dim, opt.gen_hid_dim, opt.cuda)
    discriminator = Discriminator(opt.num_class, opt.vocab_size, opt.dis_emb_dim,
                                  opt.dis_hid_dim, opt.cuda)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Loss and optimizer for the Generator and Discriminator
    gen_loss = GANLoss()
    gen_optimizer = optim.Adam(generator.parameters())
    dis_loss = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())

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
            print_samples(10, idx_to_word, samples)

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


if __name__ == '__main__':
    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    # Train the model
    main()
