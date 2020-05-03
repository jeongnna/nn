import numpy as np
from numpy.testing import assert_array_equal
from .activations import activations, Activation


def _check_layer_dim(dim):
    if isinstance(dim, int):
        assert dim > 0, 'Dimension must be greater than 0.'
        return (dim,)

    elif isinstance(dim, tuple):
        for d in dim:
            assert isinstance(d, int), 'All elements must be integers'
            assert d > 0, 'Dimension must be greater than 0.'
        return dim

    else:
        raise TypeError(f'Dimension must be an integer or a tuple. Current value is {dim}.')


def _check_activation(activation):
    if activation is None:
        return activations['identity']()

    elif isinstance(activation, str):
        return activations[activation]()

    elif isinstance(activation, Activation):
        return activation

    else:
        raise TypeError(f'Argument `activation` must be a string or an instance of {Activation}. Current value is {activation}.')
