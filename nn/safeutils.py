def safelyInputDimension(dim):
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
