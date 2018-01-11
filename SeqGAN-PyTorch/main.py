import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from helpers import read_file, create_vocab_dict, generate_samples, train_epoch, print_flags
from loss import GANLoss
from settings import parse_arguments


def main():
    if opt.cuda is not None and opt.cuda >= 0:
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True

    if opt.positive_file is None:
        if opt.das:
            opt.positive_file = '/home/tpelsmae/TextGan/data/real.data'
            opt.vocab_file = '/home/tpelsmae/TextGan/data/vocabulary.txt'
        else:
            opt.positive_file = '../data/real.data'
            opt.vocab_file = '../data/vocabulary.txt'

    random.seed(opt.seed)
    np.random.seed(opt.seed)

    idx_to_word = create_vocab_dict(opt.vocab_file)

    # Define Networks
    generator = Generator(opt.vocab_size, opt.gen_emb_dim, opt.gen_hid_dim, opt.cuda)
    discriminator = Discriminator(opt.num_class, opt.vocab_size, opt.dis_emb_dim,
                                  opt.dis_hid_dim, opt.cuda)

    real_data = read_file(opt.positive_file, opt.seq_len)

    # real_data = utils.generate_fibonacci_batch(9984, opt.seq_len)

    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Adversarial Training
    rollout = Rollout(generator, 0.8)
    print('#####################################################')
    print('Start Adverserial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(size_average=False)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for opt.num_epochs in range(opt.num_epochs):
        # Train the generator for one step
        for it in range(1):
            samples = generator.sample(opt.batch_size, opt.seq_len)

            # Print some samples
            for i in range(10):
                print(' '.join([idx_to_word[idx] for idx in samples.data[i]]))

            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((opt.batch_size, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 5, discriminator, opt.lstm_rewards)
            rewards = Variable(torch.Tensor(rewards)).contiguous().view((-1,))
            if opt.cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            prob = generator.forward(inputs)

            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        rollout.update_params()

        for _ in range(1):
            samples = generate_samples(generator, opt.batch_size, opt.num_gen, opt.seq_len)
            dis_data_iter = DisDataIter(real_data, samples, opt.batch_size, opt.lstm_rewards)
            for _ in range(1):
                loss = train_epoch(discriminator, dis_data_iter,
                                   dis_criterion, dis_optimizer, opt.batch_size, opt.cuda, opt.lstm_rewards)
                print('Batch [%d] Loss: %f' % (opt.num_epochs, loss))


if __name__ == '__main__':
    # Process and print command line arguments
    opt = parse_arguments()
    print_flags(opt)

    # Train the model
    main()
