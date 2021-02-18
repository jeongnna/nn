from abc import ABC, abstractmethod
import numpy as np
from .layers import Flatten, Dense
from .losses import SoftmaxCrossEntropy, NegativeLogLikelihood
from .data import Dataset


class Model(ABC):

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.layers = []

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def train_batch(self, X_batch, y_batch):
        out = self._forward(X_batch)
        loss = self.loss(out, y_batch)

        dloss = self.loss.backward(out, y_batch)
        dx = self._backward(dloss, optimizer=self.optimizer)

        return loss

    def fit(self, X, y, n_epochs=10, batch_size=128, *args, **kwargs):
        ds = Dataset(data=(X, y))

        for epoch in range(n_epochs):
            for X_batch, y_batch in ds.batch(batch_size):
                loss = self.train_batch(X_batch, y_batch)

            print(f'Epoch {epoch + 1} training loss: {loss}')

    def predict(self, X, *args, **kwargs):

        return self._forward(X, *args, **kwargs)

    def _forward(self, X_batch, **kwargs):
        out = X_batch
        for layer in self.layers:
            out = layer(out)

        return out

    def _backward(self, dloss, update=True, **kwargs):
        for layer in reversed(self.layers):
            dloss = layer.backward(dloss, update=update, **kwargs)

        return dloss


class Sequential(Model):

    def __init__(self, *layers):
        super().__init__()

        for layer in layers:
            self.add_layer(layer)


class LogisticClassifier(Model):

    def __init__(self, num_features, num_categories):
        super().__init__()

        self.add_layer(Flatten())
        self.add_layer(Dense(num_features, num_categories))
        self.loss = SoftmaxCrossEntropy()
