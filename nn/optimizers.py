from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name')

    @abstractmethod
    def apply_gradients(self, var, grad, **kwargs):
        assert var.shape == grad.shape

        # Update `var` using `grad` by some methods

        return kwargs


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0., *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.momentum = momentum

    """ 아마도 삭제할 코드
    def minimize(self, dloss, layers):
        for layer in reversed(layers):
            var_list = layer.trainable_variables
            dloss, grads = layer.backward(dloss)

            assert var_list.keys() == grads.keys(), f'var_list.keys(): {var_list.keys()}, grads.keys(): {grads.keys()}'

            for var, grad in zip(var_list.values(), grads.values()):
                self.apply_gradients(var, grad)
    """

    def apply_gradients(self, var, grad, **kwargs):
        assert var.shape == grad.shape

        if self.momentum > 0.:
            # v(t+1) = momentum * v(t) - learning_rate * gradient
            # theta(t+1) = theta(t) + v(t+1)

            velocity = kwargs.get('velocity', np.zeros_like(var))

            velocity = self.momentum * velocity - self.learning_rate * grad
            var += velocity

            kwargs['velocity'] = velocity

        else:
            # x += - learning_rate * dx
            var -= self.learning_rate * grad

        return kwargs


class Adam(Optimizer):
    pass
