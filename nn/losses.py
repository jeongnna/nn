from abc import ABC, abstractmethod
import numpy as np
from .math import sigmoid, softmax
from .metrics import crossEntropy


class Loss(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class SoftmaxCrossEntropy(Loss):

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, out, labels):
        self.probs = softmax(out)

        return crossEntropy(self.probs, labels)

    def backward(self, out, labels):
        N = labels.shape[0]

        dloss = self.probs
        dloss[range(N), labels] -= 1
        dloss /= N

        return dloss


class NegativeLogLikelihood(Loss):

    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, out, labels):
        N = labels.shape[0]

        self.sigmoids = sigmoid(labels.reshape(N, 1) * out)

        return np.sum(np.log(self.sigmoids)) / N

    def backward(self, out, labels):
        N = labels.shape[0]

        h = 1 - self.sigmoids
        dloss = - (labels.reshape(N, 1) * h) / N

        return dloss
