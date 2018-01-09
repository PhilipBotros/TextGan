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
    
def generate_palindrome(length):
    # Generates a single, random palindrome number of 'length' digits.
    left = [np.random.randint(0, 10) for _ in range(math.ceil(length/2))]
    left = np.asarray(left, dtype=np.int32)
    right = np.flip(left, 0) if length % 2 == 0 else np.flip(left[:-1], 0)
    return np.concatenate((left, right))

def generate_palindrome_batch(batch_size, length):
    # Generates a batch of random palindrome numbers.
    batch = [generate_palindrome(length) for _ in range(batch_size)]
    return np.asarray(batch, np.int32)