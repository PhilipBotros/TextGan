import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sample_Z(batch_size, Z_dim):
    """Sample a batch of Z from a uniform Gaussian."""
    return np.random.uniform(-1., 1., size=[batch_size, Z_dim])


def plot(samples):
    """Plot MNIST on a grid"""
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
