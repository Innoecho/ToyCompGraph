import numpy as np
from variable import Variable


class Module:
    def __init__(self):
        self.param_list = []
        self.child_list = []

    def __call__(self, x):
        return self.forward(x)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if isinstance(value, Variable):
            if value.require_grad:
                self.param_list.append(value)
        if isinstance(value, Module):
            self.child_list.append(value)

    def forward(self, x):
        raise NotImplementedError

    def get_param(self):
        param_list = self.param_list
        for child in self.child_list:
            param_list.extend(child.get_param())
        return param_list


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Variable(np.random.random((in_features, out_features)), require_grad=True)

    def forward(self, x):
        return x * self.weight
