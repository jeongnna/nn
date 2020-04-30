import numpy as np


class Loss:

    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return

    def backward(self, *args, **kwargs):
        return


class SoftmaxCrossEntropy(Loss):

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, out, y):
        probs = self._softmax(out)
        loss = self._cross_entropy(probs, y)

        self.probs = probs

        return loss

    def backward(self, out, y):
        N = y.shape[0]
        dloss = self.probs
        dloss[range(N), y] -= 1
        dloss /= N

        return dloss

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        return probs

    def _cross_entropy(self, probs, y): # https://deepnotes.io/softmax-crossentropy
        N = y.shape[0]
        log_likelihood = -np.log(probs[range(N), y])

        return np.sum(log_likelihood) / N


class NegativeLogLikelihood(Loss):

    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, out, y):
        N = y.shape[0]

        sigmoids = self._sigmoid(y.reshape(N, 1) * out)
        self.sigmoids = sigmoids

        loss = np.sum(np.log(sigmoids)) / N

        return loss

    def backward(self, out, y):
        N = y.shape[0]

        h = 1 - self.sigmoids
        dloss = - (y.reshape(N, 1) * h) / N

        return dloss

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
