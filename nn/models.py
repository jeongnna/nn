from abc import ABC, abstractmethod
import numpy as np
from .layers import Flatten, Dense
from .losses import SoftmaxCrossEntropy, NegativeLogLikelihood
from .data import Dataset


class Model(ABC):

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        pass


class LogisticClassifier(Model):

    def __init__(self, num_features, num_categories):
        self.flatten = Flatten()
        self.dense = Dense(num_features, num_categories)
        self.loss = SoftmaxCrossEntropy()

        #self.dense = Dense(num_features, 1)
        #self.loss = NegativeLogLikelihood()

    def fit(self, X, y, n_epochs=10, batch_size=128):
        #y = 2 * y - 1 # [0, 1] -> [-1, 1] # when using NegativeLogLikelihood loss

        ds = Dataset(data=(X, y))

        for epoch in range(n_epochs):
            for X_batch, y_batch in ds.batch(batch_size):
                # forward
                X_batch = self.flatten(X_batch)
                out = self.dense(X_batch)
                loss = self.loss(out, y_batch)

                # backward
                dloss = self.loss.backward(out, y_batch)
                _ = self.dense.backward(dloss)

            print(f'Epoch {epoch + 1} training loss: {loss}')

    def train(self, X, y):
        pass

    def predict(self, X):
        X = self.flatten(X)

        return self.dense(X)
