# Full Conectted
import numpy as np


def weight_init(shape):
    #return np.zeros((shape))
    return np.random.random(size=shape)-0.5
def bias_init(shape):
    return np.zeros(shape)


class Layer:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, x):
        raise NotImplementedError
    def step(self, x):
        pass


class Liner(Layer):
    def __init__(self, shape):
        assert len(shape) == 2
        self.w = weight_init(shape)
        self.b = bias_init(shape[-1])

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dt):
        self.dw = self.x.T @ dt
        self.db = dt.sum(axis=0)
        return dt @ self.w.T

    def step(self, optimf):
        self.w = optimf(self.w, self.dw)
        self.b = optimf(self.b, self.db)


class ReLu(Layer):
    def forward(self, x):
        self.mask = x >= 0
        return x * self.mask

    def backward(self, dt):
        return dt * self.mask


class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return (1 - self.forward(x)) * self.forward(x)


class MSE:
    def __call__(self, x, t):
        assert x.shape == t.shape
        loss = ((x - t) ** 2).mean()
        dt = (x - t)
        return loss, dt


class SoftMax(Layer):
    def forward(self, x):
        return np.exp(x) / (np.exp(x).sum(axis=-1).reshape(-1, 1))

    def backward(self, x):
        return (1 - self.forward(x)) * self.forward(x)


class SoftwaxCrossEntropy:
    def __init__(self):
        self.softmax = SoftMax()

    def __call__(self, x, t):
        x_softmax=self.softmax.forward(x)
        loss = -np.log((x_softmax * t).sum(axis=-1)).sum() / x.shape[0]
        dt = (x_softmax - t) / x.shape[0]
        return loss, dt


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def __call__(self, w, dw):
        assert w.shape == dw.shape
        return w - self.lr * dw


class AdaGrad:
    def __init__(self, eps=1e-8, eta=1e-3):
        self.eta = eta
        self.h = eps

    def __call__(self, w, dw):
        self.h += (dw ** 2).sum()
        self.eta /= np.sqrt(self.h)
        return w - self.eta * dw


# fc=Liner((3,2))
# fc.forward(np.array([[2,3,4],[3,4,5]]))
# fc.backward(np.array([[2,3],[4,5]]))
class Network:
    def __init__(self):
        self.layers = []
        self.layers.append(Liner((784, 10)))
        #self.layers.append(ReLu())
        #self.layers.append(Liner((512, 256)))
        #self.layers.append(ReLu())
        #self.layers.append(Liner((256, 32)))
        #self.layers.append(ReLu())
        #self.layers.append(Liner((32, 10)))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dt):
        for layer in self.layers[::-1]:
            dt = layer.backward(dt)

    def step(self, optimf):
        for layer in self.layers:
            layer.step(optimf)


# load data
numepoch = 100
batchsize = 128
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

train_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('../data', train=True, transform=Compose([
        ToTensor()])),
    batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('../data', train=False, transform=Compose([
        ToTensor(),])),
    batch_size=batchsize, shuffle=True)

net = Network()
criterion = SoftwaxCrossEntropy()
optimf = SGD()

N = 10
losslist = []
for e in range(numepoch):
    # train
    for idx, (x, y) in enumerate(train_loader):
        x = x.numpy().reshape(x.shape[0], -1)
        y = torch.eye(10)[y].numpy().reshape(y.shape[0], -1)
        output = net.forward(x)
        loss, dt = criterion(output, y)
        net.backward(dt)
        net.step(optimf)
        losslist.append(loss)
        #if idx % 100 == 0 and idx > 0: break
    print(f'Train:{e:3d},Loss:{loss:.4f}')
    # test
    totalloss = 0
    acc = 0
    acc_total=0
    for idx,(x, y) in enumerate(test_loader):
        x = x.numpy().reshape(x.shape[0], -1)
        y = y.numpy()
        output = net.forward(x)
        acc += (output.argmax(axis=-1) == y).sum()
        acc_total+=output.shape[0]
        y = np.eye(10)[y].reshape(y.shape[0], -1)
        loss, dt = criterion(output, y)
        totalloss+=loss
        # if idx%50==0 and idx>0:break
    print(f'Test:   Loss:{totalloss / len(test_loader):.4f},acc:{acc / acc_total * 100:2.2f}%')

import matplotlib.pyplot as plt

plt.plot(losslist)
plt.show()
