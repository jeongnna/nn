import numpy as np


class Dataset:

    def __init__(self, data, **kwargs):
        if isinstance(data, tuple):
            for item in data:
                assert isinstance(item, np.ndarray)
                assert len(item) == len(data[0])

        elif isinstance(data, np.ndarray):
            data = (data,)

        self.data = data

    def __iter__(self):

        return self.batch(batch_size=1)

    def __next__(self):
        ptr = self._ptr
        n = self._batch_size

        if ptr < len(self._indices):
            selected = self._indices[ptr:(ptr + n)]
            case = tuple(map(lambda x: x[selected], self.data))

            self._ptr += n

            return case

        raise StopIteration

    def __len__(self):

        return len(self.data[0])

    def batch(self, batch_size, drop_remainder=False, shuffle=False, random_seed=None):
        self._indices = np.arange(len(self))
        self._ptr = 0

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self._indices)

        self._batch_size = batch_size

        if drop_remainder:
            num_remainder = len(self._indices) % batch_size
            self._indices = self._indices[:-num_remainder]

        while self._ptr < len(self._indices):
            yield next(self)
