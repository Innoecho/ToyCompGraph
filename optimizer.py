import numpy as np


class Optimizer:
    def __init__(self, param_list):
        self.param_list = param_list

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.param_list:
            param.grad = np.zeros(param.shape)


class SGD(Optimizer):
    def __init__(self, param_list, eta=0.001):
        super().__init__(param_list)
        self.eta = eta

    def step(self):
        for param in self.param_list:
            param.data -= self.eta * param.grad


class Adam(Optimizer):
    def __init__(self, param_list, eta=0.001, beta1=0.99, beta2=0.99):
        super().__init__(param_list)
        self.eps = 1e-6
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m_list = list()
        self.s_list = list()
        for param in param_list:
            self.m_list.append(np.zeros(param.shape))
            self.s_list.append(np.zeros(param.shape))

    def step(self):
        for param, m, s in zip(self.param_list, self.m_list, self.s_list):
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            s = self.beta2 * s + (1 - self.beta2) * param.grad ** 2
            m_hat = m / (1 - self.beta1 ** (self.iter + 1))
            s_hat = s / (1 - self.beta2 ** (self.iter + 1))
            param.data -= self.eta * m_hat / np.sqrt(s_hat + self.eps)
        self.iter += 1
