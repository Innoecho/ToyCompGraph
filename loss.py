import numpy as np
from variable import Variable
from function import Function


class MSELoss(Function):
    @staticmethod
    def forward(args) -> Variable:
        result = args[0]
        target = args[1]
        err = result.data - target.data
        se_sum = np.sum(err ** 2)
        mse = se_sum / result.data.size
        loss = Variable(mse)

        return loss

    @staticmethod
    def backward(prev, grad):
        result = prev[0]
        target = prev[1]
        n = result.data.size
        a_grad = 2 * (result.data - target.data) / n
        # b_grad = -a_grad

        return [a_grad, None]
