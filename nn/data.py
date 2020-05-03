import numpy as np


class Dataset:

    def __init__(self, data):
        self.data = data # 튜플로 저장하는게 적당할 듯 (X, y) 이런 식
        # 애초에 `data`에 튜플이 입력되길 기대하지만, 만약 ndarray라면 튜플로 한번 감싸주기

        ## data 튜플 내의 array들 길이가 동일한지 검사

    def __iter__(self):

        return self.batch(batch_size=1)

    def __next__(self):
        ptr = self._ptr
        n = self._batch_size

        selected = self._indices[ptr:(ptr + n)]

        if ptr < len(self):
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

        while self._ptr < len(self):
            yield next(self)
