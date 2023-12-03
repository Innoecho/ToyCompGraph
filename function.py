import numpy as np
from variable import Variable


class Function:
    __counter = 0

    @classmethod
    def apply(cls, *args) -> Variable:
        var = cls.forward(args)
        var.set_prev(args)
        var.set_grad_fun(cls.backward)
        var.set_fun_name(cls.__name__ + f"_{cls.__counter}")
        cls.__counter += 1
        return var

    @staticmethod
    def forward(args) -> Variable:
        raise NotImplementedError

    @staticmethod
    def backward(prev, grad):
        raise NotImplementedError


class Add(Function):
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


def add(a: Variable, b: Variable):
    return Add.apply(a, b)


class Minus(Function):
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


def minus(a: Variable, b: Variable):
    return Minus.apply(a, b)


class Prod(Function):
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


def prod(a: Variable, b: Variable):
    return Prod.apply(a, b)


class Sin(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        return Variable(np.sin(a.data), is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a = prev[0]
        a_grad = np.cos(a.data) * grad
        return [a_grad]


def sin(a: Variable):
    return Sin.apply(a)


class Cos(Function):
    @staticmethod
    def forward(args) -> Variable:
        a = args[0]
        return Variable(np.cos(a.data), is_leaf=False, require_grad=True)

    @staticmethod
    def backward(prev, grad):
        a = prev[0]
        a_grad = -np.sin(a.data) * grad
        return [a_grad]


def cos(a: Variable):
    return Cos.apply(a)
