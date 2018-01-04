from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf


def generate_palindrome(length):
    # Generates a single, random palindrome number of 'length' digits.
    left = [np.random.randint(0, 10) for _ in range(int(math.ceil(length / 2)))]
    left = np.asarray(left, dtype=np.int32)
    right = np.flip(left, 0) if length % 2 == 0 else np.flip(left[:-1], 0)
    return np.concatenate((left, right))


def generate_palindrome_batch(batch_size, length):
    # Generates a batch of random palindrome numbers.
    batch = [generate_palindrome(length) for _ in range(batch_size)]
    return np.asarray(batch, np.int32)


def generate_fibonacci_batch(batch_size, length):
    # Generates a batch of random palindrome numbers.
    batch = [generate_fibonacci(length) for _ in range(batch_size)]
    return np.asarray(batch, np.int32)


def generate_fibonacci(length):
    fib = [np.random.randint(0, 10) for _ in range(2)]
    for i in range(length - 2):
        fib.append(fib[i] + fib[i + 1])
    return np.asarray(fib, dtype=np.int32)
