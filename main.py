import numpy as np
import matplotlib.pyplot as plt

from variable import Variable
from loss import MSELoss
from optimizer import Adam


class Sample:
    def __init__(self, dim=4):
        self.__dim = dim
        self.__param = np.random.random((dim, 1))

    def get_param(self):
        return self.__param

    def get_batch(self, n):
        data_ = np.zeros((n, self.__dim))
        target_ = np.zeros((n, 1))
        for i in range(n):
            x = (np.random.random() - 0.5) * 2 * 10
            for j in range(self.__dim):
                data_[i, j] = x ** j
            target_[i] = np.dot(data_[i, :], self.__param)
        return Variable(data_), Variable(target_)


class Model:
    def __init__(self):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError


class SimpleModel(Model):
    def __init__(self, dim):
        super().__init__()
        self.param = Variable(np.random.random((dim, 1)))

    def forward(self, input):
        return input * self.param

    def get_param(self):
        return [self.param]


dim = 3

iter_num = 5000
batch_size = 100

eta = 0.01
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
x, target = sample.get_batch(valid_batch_size)
y = model(x)

plt.subplot(121)
plt.plot(loss_history)
plt.subplot(122)
plt.plot(data_history[:, 0].T, data_history[:, 1].T, 'k.', x.data[:, 1], y.data, 'r.')

plt.show()

# view_graph(loss, label_type=-1)
