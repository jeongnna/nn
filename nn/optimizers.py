from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name')

    @abstractmethod
    def apply_gradients(self, var, grad, **kwargs):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.momentum = momentum

    def minimize(self, dloss, layers):
        for layer in reversed(layers):
            var_list = layer.trainable_variables
            dloss, grads = layer.backward(dloss)

            assert var_list.keys() == grads.keys(), f'var_list.keys(): {var_list.keys()}, grads.keys(): {grads.keys()}'

            for var, grad in zip(var_list.values(), grads.values()):
                self.apply_gradients(var, grad)

    def apply_gradients(self, var, grad):
        assert var.shape == grad.shape

        if self.momentum:
            # v(t+1) = momentum * v(t) - learning_rate * gradient
            # theta(t+1) = theta(t) + v(t+1)
            velocity = self.velocitys[id(var)] # id(var) 대체할 var 고유의 무언가가 있을까

            velocity = self.momentum * velocity - self.learning_rate * grad
            var += velocity

            self.velocitys[id(var)] = velocitys
        else:
            # x += - learning_rate * dx
            var -= self.learning_rate * grad



class Adam(Optimizer):
    pass
