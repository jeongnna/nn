from abc import ABC, abstractmethod
import numpy as np
from .layers import Dense
from .losses import SoftmaxCrossEntropy, NegativeLogLikelihood


class Model(ABC):

    def __init__(self):
        return

    @abstractmethod
    def fit(self, X, y, *args):
        return

    @abstractmethod
    def predict(self, X, *args):
        return


class LogisticClassifier(Model):

    def __init__(self, num_features, num_category):
        super(LogisticClassifier, self).__init__()

        self.dense = Dense(num_features, num_category)
        self.loss = SoftmaxCrossEntropy()

        #self.dense = Dense(num_features, 1)
        #self.loss = NegativeLogLikelihood()

    def fit(self, X, y, n_iter=10):
        #y = 2 * y - 1 # [0, 1] -> [-1, 1] # when using NegativeLogLikelihood loss

        for _ in range(n_iter):
            # forward
            out = self.dense(X)
            loss = self.loss(out, y)
            print('loss: {}'.format(loss))

            # backward
            dloss = self.loss.backward(out, y)
            _ = self.dense.backward(dloss)

    def predict(self, X):
        return self.dense(X)
