import numpy as np


def topo_sort(node):
    node_visited = set()
    topo_sorted = []

    def topo_sort_helper(node_):
        for prev_node in node_.get_prev():
            if prev_node not in node_visited:
                node_visited.add(prev_node)
                topo_sort_helper(prev_node)
            topo_sorted.append(node_)

    topo_sort_helper(node)
    return reversed(topo_sorted)


class Variable:
    __counter = 0

    def __init__(self, data: np.ndarray, is_leaf=True, require_grad=False):
        assert len(data.shape) <= 2, 'just 2-dim ndarray be supported'
        self._id = Variable.__counter
        Variable.__counter += 1

        self._func_name = None
        self._name = self.__class__.__name__ + f"_{self.__class__.__counter}"

        self.is_leaf = is_leaf
        self.require_grad = require_grad

        self._prev = []
        self._grad_fun = None

        self.shape = data.shape
        self.data = data
        self.grad = np.ones(self.shape)

    def set_prev(self, prev):
        self._prev = prev

    def get_prev(self):
        return self._prev

    def set_grad_fun(self, grad_fun):
        self._grad_fun = grad_fun

    def set_fun_name(self, fun_name):
        self._func_name = fun_name

    def get_name(self) -> str:
        if self.is_leaf:
            return self._name
        else:
            return self._func_name

    def get_label(self, label_type: int) -> str:
        if label_type == 1:
            return self.get_name() + f": {self.data.flatten()[0]:.3f}"
        elif label_type == -1:
            return self.get_name() + f": {self.grad.flatten()[0]:.3f}"
        else:
            return self.get_name()

    def backward(self):
        topo_sorted = topo_sort(self)

        for var in topo_sorted:
            var.exec_backward()

    def exec_backward(self):
        if self.is_leaf:
            return
        grad = self._grad_fun(self._prev, self.grad)
        for i, v in enumerate(grad):
            if self._prev[i].require_grad and v:
                self._prev[i].grad += v

    def __add__(self, var):
        from function import add
        return add(self, var)

    def __sub__(self, var):
        from function import minus
        return minus(self, var)

    def __mul__(self, var):
        from function import prod
        return prod(self, var)
