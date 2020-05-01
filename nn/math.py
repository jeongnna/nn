import numpy as np


def sigmoid(x):
    #return 1. / (1. + np.exp(-x))
    return np.exp(np.minimum(0, x)) / (1. + np.exp(-np.abs(x)))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    return probs
