import numpy as np
from variable import Variable


class Function:
    __counter = 0

    def __call__(self, *args) -> Variable:
        var = self.__class__.forward(args)
        var.set_prev(args)
        var.set_grad_fun(self.__class__.backward)
        var.set_fun_name(self.__class__.__name__[:-4] + f"_{self.__class__.__counter}")
        self.__class__.__counter += 1
        return var

    @staticmethod
    def forward(args) -> Variable:
        raise NotImplementedError

    @staticmethod
    def backward(prev, grad):
        raise NotImplementedError


class AddMeta(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        b = args[1]
        assert a.shape == b.shape
        return Variable(a.data + b.data, is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a_grad = grad
        b_grad = grad
        return [a_grad, b_grad]


class MinusMeta(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        b = args[1]
        assert a.shape == b.shape
        return Variable(a.data - b.data, is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a_grad = grad
        b_grad = -grad
        return [a_grad, b_grad]


class ProdMeta(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        b = args[1]
        assert a.shape[-1] == b.shape[0]
        return Variable(np.dot(a.data, b.data), is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a = prev[0]
        b = prev[1]
        a_grad = np.dot(grad, b.data.T)
        b_grad = np.dot(a.data.T, grad)
        return [a_grad, b_grad]


class SinMeta(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        return Variable(np.sin(a.data), is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a = prev[0]
        a_grad = np.cos(a.data) * grad
        return [a_grad]


class CosMeta(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        return Variable(np.cos(a.data), is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a = prev[0]
        a_grad = -np.sin(a.data) * grad
        return [a_grad]


add = AddMeta()
minus = MinusMeta()
prod = ProdMeta()
sin = SinMeta()
cos = CosMeta()
