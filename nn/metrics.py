import numpy as np


def cross_entropy(probs, y): # https://deepnotes.io/softmax-crossentropy
    N = y.shape[0]

    log_likelihood = -np.log(probs[range(N), y])

    return np.sum(log_likelihood) / N
