from variable import Variable


class Model:
    def __init__(self):
        self.param_list = []

    def __call__(self, x):
        return self.forward(x)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if isinstance(value, Variable):
            if value.require_grad:
                self.param_list.append(value)

    def forward(self, x):
        raise NotImplementedError

    def get_param(self):
        return self.param_list
