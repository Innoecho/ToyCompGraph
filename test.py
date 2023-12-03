import numpy as np
import matplotlib.pyplot as plt

from variable import Variable
from loss import MSELoss
from optimizer import Adam

from sample import Sample
from model import Model


class SimpleModel(Model):
    def __init__(self, dim_):
        super().__init__()
        self.param_list = []
        self.param = Variable(np.random.random((dim_, 1)), require_grad=True)

    def forward(self, x):
        return x * self.param


if __name__ == '__main__':
    dim = 3

    iter_num = 50
    batch_size = 100

    eta = 0.1
    beta1 = 0.99
    beta2 = 0.99

    loss_history = np.zeros((iter_num, 1))
    data_history = np.zeros((iter_num * batch_size, 2))

    sample = Sample(dim)
    mse = MSELoss()
    model = SimpleModel(dim)
    optimizer = Adam(model.get_param(), eta=eta, beta1=beta1, beta2=beta2)

    for i_iter in range(iter_num):
        data, target = sample.get_batch(batch_size)

        optimizer.zero_grad()
        output = model(data)
        loss = mse(output, target)
        loss.backward()
        optimizer.step()

        loss_history[i_iter] = loss.data
        data_history[i_iter * batch_size: (i_iter + 1) * batch_size, 0] = data.data[:, 1]
        data_history[i_iter * batch_size: (i_iter + 1) * batch_size, 1] = target.data[:, 0]

    valid_batch_size = 100
    data, target = sample.get_batch(valid_batch_size)
    output = model(data)

    plt.subplot(121)
    plt.plot(loss_history)
    plt.subplot(122)
    plt.plot(data_history[:, 0].T, data_history[:, 1].T, 'k.', data.data[:, 1], output.data, 'r.')

    plt.show()
