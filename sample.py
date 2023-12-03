import numpy as np
from variable import Variable


class Sample:
    def __init__(self, dim=4):
        self.__dim = dim
        self.__param = np.random.random((dim, 1))

    def get_param(self):
        return self.__param

    def get_batch(self, n):
        data_ = np.ones((n, self.__dim))
        x = (np.random.random(size=(n, 1)) - 0.5) * 2 * 10
        for j in range(1, self.__dim):
            data_[:, j:j+1] = x ** j
        target_ = np.dot(data_, self.__param)
        return Variable(data_), Variable(target_)
